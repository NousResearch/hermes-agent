import { QueryClient } from '@tanstack/react-query'
import { act, cleanup, render, waitFor } from '@testing-library/react'
import { useEffect, useRef } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ClientSessionState } from '@/app/types'
import { chatMessageText } from '@/lib/chat-messages'
import { createClientSessionState } from '@/lib/chat-runtime'
import { clearSessionTodos } from '@/store/todos'
import type { RpcEvent } from '@/types/hermes'

import { useMessageStream } from './index'

const SID = 'session-1'

let handleEvent: ((event: RpcEvent) => void) | null = null
let sessionStates: Map<string, ClientSessionState>
let mockCompleteSound: ReturnType<typeof vi.fn>
let mockHaptic: ReturnType<typeof vi.fn>

function Harness() {
  const activeSessionIdRef = useRef<string | null>(SID)
  const sessionStateByRuntimeIdRef = useRef(new Map<string, ClientSessionState>())
  const queryClientRef = useRef(new QueryClient())

  const stream = useMessageStream({
    activeSessionIdRef,
    hydrateFromStoredSession: vi.fn(async () => undefined),
    queryClient: queryClientRef.current,
    refreshHermesConfig: vi.fn(async () => undefined),
    refreshSessions: vi.fn(async () => undefined),
    sessionStateByRuntimeIdRef,
    updateSessionState: (sessionId, updater) => {
      const current = sessionStateByRuntimeIdRef.current.get(sessionId) ?? createClientSessionState()
      const next = updater(current)
      sessionStateByRuntimeIdRef.current.set(sessionId, next)
      sessionStates.set(sessionId, next)

      return next
    }
  })

  useEffect(() => {
    handleEvent = stream.handleGatewayEvent
  }, [stream.handleGatewayEvent])

  return null
}

async function mountStream() {
  sessionStates = new Map()
  render(<Harness />)
  await waitFor(() => expect(handleEvent).not.toBeNull())
}

const start = () => act(() => handleEvent!({ payload: {}, session_id: SID, type: 'message.start' }))
const delta = (text: string) => act(() => handleEvent!({ payload: { text }, session_id: SID, type: 'message.delta' }))

const interim = (text: string) =>
  act(() => handleEvent!({ payload: { text, already_streamed: true }, session_id: SID, type: 'message.interim' }))

const complete = (text: string) =>
  act(() => handleEvent!({ payload: { text }, session_id: SID, type: 'message.complete' }))

const completePreviewed = (text: string) =>
  act(() => handleEvent!({ payload: { text, response_previewed: true }, session_id: SID, type: 'message.complete' }))

function getState(): ClientSessionState {
  return sessionStates.get(SID) ?? createClientSessionState()
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
    handleEvent = null
    clearSessionTodos(SID)
  })

  afterEach(() => {
    cleanup()
    clearSessionTodos(SID)
    vi.restoreAllMocks()
  })

  it('preserves interim text that the final response does not include', async () => {
    await mountStream()
    await start()

    await delta('awaaaaa clean!! tsc zero errors')
    await interim('awaaaaa clean!! tsc zero errors')

    await complete('All checks passed.')

    const texts = assistantMessages()
    expect(texts).toContain('awaaaaa clean!! tsc zero errors')
    expect(texts).toContain('All checks passed.')
  })

  it('marks sealed interim bubbles interim and leaves the final reply unmarked', async () => {
    await mountStream()
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
    await mountStream()
    await start()

    await interim('same reply')
    await completePreviewed('same reply')

    const assistants = getState().messages.filter(m => m.role === 'assistant' && !m.hidden)
    expect(assistants).toHaveLength(1)
    expect(assistants[0].interim).toBeFalsy()
  })

  it('dedupes interim text when the final response includes it', async () => {
    await mountStream()
    await start()

    await delta('Let me check the files.')
    await interim('Let me check the files.')

    await complete('Let me check the files. Everything looks good.')

    const texts = assistantMessages()
    expect(texts).not.toContain('Let me check the files.Let me check the files.')
    expect(texts.some(t => t.includes('Let me check the files. Everything looks good.'))).toBe(true)
  })

  it('clears interimBoundaryPending at turn end so the next turn starts clean', async () => {
    await mountStream()
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
    await mountStream()
    await start()

    await delta('streaming text')
    await interim('streaming text')

    // Turn is still active — busy stays true
    expect(getState().busy).toBe(true)
    expect(getState().interimBoundaryPending).toBe(true)
  })

  it('keeps an identical final completion distinct from an interim reply without response_previewed', async () => {
    await mountStream()
    await start()

    await interim('same reply')
    await complete('same reply')

    // Without response_previewed, the interim and terminal replies are
    // distinct messages — the gateway didn't signal that the final reuses
    // the provisional candidate.
    const texts = assistantMessages()
    expect(texts.filter(t => t === 'same reply')).toHaveLength(2)
  })

  it('settles an identical final completion onto the interim when response_previewed', async () => {
    await mountStream()
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
    await mountStream()
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
    await mountStream()
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
    await mountStream()
    await start()

    // No payload at all
    await act(() => handleEvent!({ type: 'message.interim' } as RpcEvent))
    // Empty text
    await act(() => handleEvent!({ payload: { text: '' }, session_id: SID, type: 'message.interim' } as RpcEvent))
    // Undefined text
    await act(() =>
      handleEvent!({ payload: { text: undefined }, session_id: SID, type: 'message.interim' } as RpcEvent)
    )

    // Turn continues without finalizing or throwing
    expect(getState().busy).toBe(true)
    expect(getState().interimBoundaryPending).toBe(false)
  })

  it('clears interimBoundaryPending on message.start', async () => {
    await mountStream()
    await start()

    await delta('interim text')
    await interim('interim text')
    expect(getState().interimBoundaryPending).toBe(true)

    // New turn starts
    await start()
    expect(getState().interimBoundaryPending).toBe(false)
  })
})
