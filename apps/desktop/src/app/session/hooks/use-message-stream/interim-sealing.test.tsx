import { act, cleanup } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ClientSessionState } from '@/app/types'
import { chatMessageText, textPart } from '@/lib/chat-messages'
import { createClientSessionState } from '@/lib/chat-runtime'
import { clearSessionTodos } from '@/store/todos'
import type { RpcEvent } from '@/types/hermes'

import { type MessageStreamHarness, renderMessageStream } from './test-harness'

const SID = 'session-1'

let stream: MessageStreamHarness
let mockCompleteSound: ReturnType<typeof vi.fn>
let mockHaptic: ReturnType<typeof vi.fn>

function mountStream() {
  stream = renderMessageStream(SID)
}

const start = () => act(() => stream.handleEvent({ payload: {}, session_id: SID, type: 'message.start' }))

const delta = (text: string) =>
  act(() => stream.handleEvent({ payload: { text }, session_id: SID, type: 'message.delta' }))

const interim = (text: string) =>
  act(() => stream.handleEvent({ payload: { text, already_streamed: true }, session_id: SID, type: 'message.interim' }))

const complete = (text: string) =>
  act(() => stream.handleEvent({ payload: { text }, session_id: SID, type: 'message.complete' }))

const completePreviewed = (text: string) =>
  act(() =>
    stream.handleEvent({ payload: { text, response_previewed: true }, session_id: SID, type: 'message.complete' })
  )

const toolStart = (toolId = 'tool-1') =>
  act(() =>
    stream.handleEvent({
      payload: { args: { command: 'echo ok' }, name: 'terminal', tool_id: toolId },
      session_id: SID,
      type: 'tool.start'
    })
  )

function getState(): ClientSessionState {
  return stream.state()
}

function assistantText(): string {
  const state = getState()
  const last = [...state.messages].reverse().find(m => m.role === 'assistant' && !m.hidden)

  return last ? chatMessageText(last) : ''
}

function assistantMessages(): string[] {
  const state = getState()

  return state.messages
    .filter(m => m.role === 'assistant' && !m.hidden)
    .map(m => chatMessageText(m))
    .filter(Boolean)
}

describe('useMessageStream interim text sealing', () => {
  beforeEach(() => {
    clearSessionTodos(SID)
  })

  afterEach(() => {
    cleanup()
    clearSessionTodos(SID)
    vi.restoreAllMocks()
  })

  it('preserves interim text that the final response does not include', async () => {
    mountStream()
    await start()

    await delta('awaaaaa clean!! tsc zero errors')
    await interim('awaaaaa clean!! tsc zero errors')

    await complete('All checks passed.')

    const texts = assistantMessages()
    expect(texts).toContain('awaaaaa clean!! tsc zero errors')
    expect(texts).toContain('All checks passed.')
  })

  it('hydrates an empty completion when the live transcript still ends with the user message', async () => {
    // #88036: the terminal frame arrived empty while this window never
    // rendered any assistant output — the reply exists only in stored
    // history, so the settle must hydrate instead of leaving the
    // transcript ending on the user's message until restart.
    const hydrateFromStoredSession = vi.fn<() => Promise<void>>(async () => undefined)

    stream = renderMessageStream(SID, {
      hydrateFromStoredSession,
      states: new Map([
        [
          SID,
          createClientSessionState('stored-session-1', [
            { id: 'user-1', parts: [textPart('finish the task')], role: 'user' }
          ])
        ]
      ])
    })
    await start()

    await complete('')

    expect(hydrateFromStoredSession).toHaveBeenCalledWith(3, 'stored-session-1', SID)
  })

  it('marks sealed interim bubbles interim and leaves the final reply unmarked', async () => {
    mountStream()
    await start()

    await delta('Let me check the files.')
    await interim('Let me check the files.')
    await delta('Now the second pass.')
    await interim('Now the second pass.')
    await complete('All done.')

    const assistants = getState().messages.filter(m => m.role === 'assistant' && !m.hidden)
    const byText = (text: string) => assistants.find(m => chatMessageText(m) === text)

    expect(byText('Let me check the files.')?.interim).toBe(true)
    expect(byText('Now the second pass.')?.interim).toBe(true)
    expect(byText('All done.')?.interim).toBeFalsy()
  })

  it('clears the interim mark when a previewed final settles onto the interim bubble', async () => {
    mountStream()
    await start()

    await interim('same reply')
    await completePreviewed('same reply')

    const assistants = getState().messages.filter(m => m.role === 'assistant' && !m.hidden)
    expect(assistants).toHaveLength(1)
    expect(assistants[0].interim).toBeFalsy()
  })

  it('dedupes interim text when the final response includes it', async () => {
    mountStream()
    await start()

    await delta('Let me check the files.')
    await interim('Let me check the files.')

    await complete('Let me check the files. Everything looks good.')

    const texts = assistantMessages()
    expect(texts).not.toContain('Let me check the files.Let me check the files.')
    expect(texts.some(t => t.includes('Let me check the files. Everything looks good.'))).toBe(true)
  })

  it('clears interimBoundaryPending at turn end so the next turn starts clean', async () => {
    mountStream()
    await start()

    await delta('interim text')
    await interim('interim text')
    await complete('final text')

    expect(getState().interimBoundaryPending).toBe(false)

    await start()
    expect(getState().interimBoundaryPending).toBe(false)

    await complete('new turn final')

    const texts = assistantMessages()
    expect(texts[texts.length - 1]).toBe('new turn final')
  })

  it('finalizes an interim segment without settling the turn', async () => {
    mountStream()
    await start()

    await delta('streaming text')
    await interim('streaming text')

    // Turn is still active — busy stays true
    expect(getState().busy).toBe(true)
    expect(getState().interimBoundaryPending).toBe(true)
  })

  it('settles an identical final onto a non-previewed interim (tool-call turn) instead of duplicating (#63679)', async () => {
    mountStream()
    await start()

    // A plain tool-call turn: the streamed text is sealed as an interim at the
    // tool boundary (no response_previewed — that flag is only for verify-on-
    // stop). The final completion is the SAME turn's reply. It must settle onto
    // the interim, not append a second bubble — the DB has one row. This is the
    // "renders twice" bug: partial streamed copy + clean final copy side by side.
    await interim('same reply')
    await complete('same reply')

    const texts = assistantMessages()
    expect(texts.filter(t => t === 'same reply')).toHaveLength(1)
  })

  it('settles an identical final after the tool row that follows its interim (#98524)', async () => {
    mountStream()
    await start()

    // Production order: message.interim seals the narrated response and clears
    // streamId, then tool.start creates the next live assistant row. Completion
    // must not paint the same authoritative final onto that row while retaining
    // the identical sealed interim a few rows earlier.
    await delta('same reply')
    await interim('same reply')
    await toolStart()
    await complete('same reply')

    const assistants = getState().messages.filter(m => m.role === 'assistant' && !m.hidden)
    const texts = assistants.map(m => chatMessageText(m)).filter(Boolean)

    expect(texts.filter(text => text === 'same reply')).toHaveLength(1)
    expect(assistants.flatMap(message => message.parts).filter(part => part.type === 'tool-call')).toHaveLength(1)
  })

  it('preserves distinct tool calls when a provider reuses ids across responses', async () => {
    mountStream()
    await start()

    await delta('same reply')
    await toolStart('terminal_0')
    await interim('same reply')
    await toolStart('terminal_0')
    await complete('same reply')

    const tools = getState()
      .messages.flatMap(message => message.parts)
      .filter(part => part.type === 'tool-call')

    expect(tools).toHaveLength(2)
    expect(new Set(tools.map(tool => tool.toolCallId)).size).toBe(2)
  })

  it('keeps distinct pre-tool commentary and final text as separate assistant segments', async () => {
    mountStream()
    await start()

    await interim('Let me inspect that.')
    await toolStart()
    await complete('The implementation is correct.')

    expect(assistantMessages()).toEqual(['Let me inspect that.', 'The implementation is correct.'])
    expect(
      getState()
        .messages.flatMap(message => message.parts)
        .filter(part => part.type === 'tool-call')
    ).toHaveLength(1)
  })

  it('settles a prefix-extended final onto a non-previewed interim (streamed + trailing delta)', async () => {
    mountStream()
    await start()

    // The stream dropped/settled early at the tool boundary; the final adds a
    // trailing delta. Same turn — one bubble with the full final text.
    await delta('partial')
    await interim('partial')
    await complete('partial answer continued')

    const texts = assistantMessages()
    expect(texts.filter(t => t.includes('partial'))).toHaveLength(1)
    expect(texts[0]).toBe('partial answer continued')
  })

  it('settles final onto interim even after message.start reset the boundary flag (#74560)', async () => {
    mountStream()
    await start()
    await delta('partial')
    await interim('partial')
    // A chained turn / follow-up re-emits message.start, which resets
    // interimBoundaryPending to false BEFORE the same turn's message.complete.
    // The continuation must still settle onto the interim — not append a
    // duplicate bubble. Regression for #74560.
    await start()
    await complete('partial answer continued')

    const texts = assistantMessages()
    expect(texts.filter(t => t.includes('partial'))).toHaveLength(1)
    expect(texts[0]).toBe('partial answer continued')
  })

  it('appends a distinct previewed final after a message.start reset instead of overwriting the interim', async () => {
    mountStream()
    await start()
    await interim('old interim text')
    // A genuinely new turn begins — message.start resets interimBoundaryPending.
    // Production ordering: mid-turn compaction-resume events do NOT include
    // message.start (COMPACTION_RESUME_EVENT_TYPES in gateway-event/index.ts), and
    // the TUI gateway emits message.complete BEFORE goal-followup starts
    // (tui_gateway/server.py), so a previewed final arriving after the reset
    // is a DISTINCT reply, not a rewrite of the interim. It must append its
    // own bubble — never overwrite the old one (sweeper review on #76583).
    await start()
    await completePreviewed('totally new answer')

    const texts = assistantMessages()
    expect(texts).toContain('old interim text')
    expect(texts).toContain('totally new answer')
    expect(texts).toHaveLength(2)
  })

  it('appends a genuinely different final as its own bubble (two real assistant segments)', async () => {
    mountStream()
    await start()

    // The interim is one segment (pre-tool commentary); the final is different
    // content, not a continuation of it. These are two real messages and must
    // both render — the fix must not over-collapse distinct replies.
    await interim('let me check the files')
    await complete('the answer is 42')

    const texts = assistantMessages()
    expect(texts).toContain('let me check the files')
    expect(texts).toContain('the answer is 42')
    expect(texts).toHaveLength(2)
  })

  it('settles an identical final completion onto the interim when response_previewed', async () => {
    mountStream()
    await start()

    await interim('same reply')
    await completePreviewed('same reply')

    // With response_previewed, the final text is the same model response
    // that was published provisionally as an interim — settle onto the
    // existing interim instead of creating a duplicate. (#65919 review)
    const texts = assistantMessages()
    expect(texts.filter(t => t === 'same reply')).toHaveLength(1)
  })

  it('settles a prefix-matched final onto the interim when response_previewed', async () => {
    mountStream()
    await start()

    // Interim text is a PREFIX of the final — the model streamed part of
    // its answer before the verify nudge fired, then the final includes
    // the same text plus a trailing delta.
    await interim('partial answer')
    await completePreviewed('partial answer with more detail')

    // Prefix match: the final starts with the interim text, so settle
    // onto the interim instead of creating a duplicate bubble.
    const texts = assistantMessages()
    expect(texts.filter(t => t.includes('partial answer'))).toHaveLength(1)
    expect(texts[0]).toBe('partial answer with more detail')
  })

  it('dedupes partial-stream-then-nudge: streamed prefix + interim + previewed final settles to one bubble', async () => {
    mountStream()
    await start()

    // The model streamed part of its answer via message.delta, then the
    // verify nudge fired. The interim seals the streamed text, then the
    // final response is the same text plus a trailing delta.
    await delta('partial streamed')
    await interim('partial streamed')
    await completePreviewed('partial streamed answer continued')

    // One bubble, containing the full final text — not two.
    const texts = assistantMessages()
    expect(texts.filter(t => t.includes('partial streamed'))).toHaveLength(1)
    expect(texts[0]).toBe('partial streamed answer continued')
  })

  it('ignores malformed message.interim payload', async () => {
    mountStream()
    await start()

    // No payload at all
    await act(() => stream.handleEvent({ type: 'message.interim' } as RpcEvent))
    // Empty text
    await act(() => stream.handleEvent({ payload: { text: '' }, session_id: SID, type: 'message.interim' } as RpcEvent))
    // Undefined text
    await act(() =>
      stream.handleEvent({ payload: { text: undefined }, session_id: SID, type: 'message.interim' } as RpcEvent)
    )

    // Turn continues without finalizing or throwing
    expect(getState().busy).toBe(true)
    expect(getState().interimBoundaryPending).toBe(false)
  })

  it('clears interimBoundaryPending on message.start', async () => {
    mountStream()
    await start()

    await delta('interim text')
    await interim('interim text')
    expect(getState().interimBoundaryPending).toBe(true)

    // New turn starts
    await start()
    expect(getState().interimBoundaryPending).toBe(false)
  })
})
