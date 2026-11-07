import type { GatewayEvent } from '@hermes/shared'
// Repro for "when I steer it often sends out of order — a user bubble way
// above" (#73793 / #83151 class). Drives the REAL stream reducer
// (useMessageStream.handleGatewayEvent) and the REAL steer entry point
// (usePromptActions.redirectPrompt — appendAfterActiveReply guard, optimistic
// insert, streamId seal) through a full steered turn, and asserts transcript
// ORDER:
//
//   pre-steer output → steer bubble → post-steer output → settled reply
//
// The steer bubble must land AFTER every assistant row that had already
// streamed when it was typed, and every later delta / tool event / completion
// must land BELOW it — never spliced above, never merged into the sealed
// pre-steer bubble.
//
// Both hooks share one state map, exactly as the desktop wires them: steering
// mutates the same ClientSessionState the gateway events reduce into.
import { QueryClient } from '@tanstack/react-query'
import { act, cleanup, render, renderHook } from '@testing-library/react'
import { useEffect, useRef } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { MAIN_COMPOSER_SCOPE } from '@/app/chat/composer/scope'
import { useRuntimeMessageRepository } from '@/app/chat/runtime-repository'
import { useSessionTileActions } from '@/app/chat/session-tile-actions'
import { usePromptActions } from '@/app/session/hooks/use-prompt-actions'
import type { ClientSessionState } from '@/app/types'
import { type ChatMessage, chatMessageText } from '@/lib/chat-messages'
import { createClientSessionState } from '@/lib/chat-runtime'
import { IncrementalExternalStoreRuntimeCore } from '@/lib/incremental-external-store-runtime'
import { setSessionTileDelegate } from '@/store/session-states'

import { STREAM_DELTA_FLUSH_MS } from './utils'

import { useMessageStream } from './index'

const SID = 'steer-order-session'

let handleEvent: ((event: GatewayEvent) => void) | null = null
let redirect: ((text: string, mode?: 'interrupt' | 'steer') => Promise<boolean>) | null = null
let states: Map<string, ClientSessionState>

/** Stub only the RPC boundary; both action handlers and streaming reducers are real. */
const requestGatewayMock = vi.fn(async (method: string): Promise<unknown> =>
  method === 'session.redirect' ? { status: 'redirected' } : {}
)

const requestGateway = requestGatewayMock as unknown as <T>(
  method: string,
  params?: Record<string, unknown>,
  timeoutMs?: number
) => Promise<T>

function Harness({ tile = false }: { tile?: boolean }) {
  const activeSessionIdRef = useRef<null | string>(SID)
  const sessionStateByRuntimeIdRef = useRef(new Map<string, ClientSessionState>())
  const queryClientRef = useRef(new QueryClient())
  const busyRef = useRef(false)
  const runtimeIdByStoredSessionIdRef = useRef(new Map<string, string>())
  const selectedStoredSessionIdRef = useRef<null | string>(SID)

  const updateSessionState = (sessionId: string, updater: (state: ClientSessionState) => ClientSessionState) => {
    const current = sessionStateByRuntimeIdRef.current.get(sessionId) ?? createClientSessionState()
    const next = updater(current)
    sessionStateByRuntimeIdRef.current.set(sessionId, next)

    return next
  }

  const stream = useMessageStream({
    activeSessionIdRef,
    hydrateFromStoredSession: vi.fn(async () => undefined),
    queryClient: queryClientRef.current,
    refreshHermesConfig: vi.fn(async () => undefined),
    refreshSessions: vi.fn(async () => undefined),
    sessionStateByRuntimeIdRef,
    updateSessionState
  })

  const actions = usePromptActions({
    activeSessionId: SID,
    activeSessionIdRef,
    branchCurrentSession: async () => true,
    busyRef,
    createBackendSessionForSend: async () => SID,
    getRoutedStoredSessionId: () => null,
    getRuntimeIdForStoredSession: () => null,
    getRouteToken: () => 'token',
    handleSkinCommand: () => '',
    openMemoryGraph: () => undefined,
    refreshSessions: async () => undefined,
    requestGateway,
    resumeStoredSession: () => undefined,
    runtimeIdByStoredSessionIdRef,
    selectedStoredSessionIdRef,
    startFreshSessionDraft: () => undefined,
    sttEnabled: false,
    updateSessionState
  })

  const tileActions = useSessionTileActions({
    requestGateway,
    runtimeId: SID,
    storedSessionId: SID,
    scope: MAIN_COMPOSER_SCOPE
  })

  useEffect(() => {
    setSessionTileDelegate({
      archiveSession: async () => undefined,
      branchSession: async () => undefined,
      branchSessionAtMessage: async () => false,
      deleteSession: async () => undefined,
      executeSlash: async () => undefined,
      interruptSession: async () => undefined,
      resumeTile: async () => SID,
      submitToSession: async () => ({ runtimeSessionId: SID, storedSessionId: SID }),
      updateSession: updateSessionState
    })
    handleEvent = stream.handleGatewayEvent
    redirect = tile ? tileActions.steerPrompt : actions.redirectPrompt
    states = sessionStateByRuntimeIdRef.current
  }, [stream.handleGatewayEvent, actions.redirectPrompt, tileActions.steerPrompt, tile])

  return null
}

async function mountHarness(tile = false) {
  vi.useFakeTimers()
  render(<Harness tile={tile} />)
  await act(async () => {
    await Promise.resolve()
  })
}

const flushDeltas = async () => {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(STREAM_DELTA_FLUSH_MS)
  })
}

const emit = (event: GatewayEvent) => act(() => handleEvent?.(event))

/** A real steer: redirectPrompt's optimistic insert + the gateway round-trip. */
const steer = async (text: string) => {
  await act(async () => {
    await expect(redirect!(text)).resolves.toBe(true)
  })
}

const transcript = () =>
  (states.get(SID)?.messages ?? []).map(message => `${message.role}:${chatMessageText(message).slice(0, 30)}`)

/** Rows as painted: thinking, tool calls and text per bubble, in order. */
const layout = () =>
  (states.get(SID)?.messages ?? []).map(message => {
    const body = message.parts
      .map(part =>
        part.type === 'reasoning'
          ? `think(${part.text})`
          : part.type === 'tool-call'
            ? `tool(${part.toolCallId})`
            : part.type === 'text'
              ? part.text
              : ''
      )
      .filter(Boolean)
      .join(' + ')

    return `${message.role}:${body}`
  })

/** The `steer.delivered` payload for the submissions one drain joined. */
const delivery = (texts: string[]) => ({ count: texts.length, texts })

/** A steer the gateway accepts into the live turn and will announce with `steer.delivered`. */
const acceptSteer = async (text: string) => {
  requestGatewayMock.mockImplementationOnce(async () => ({ status: 'queued', delivery_event: true }))
  await act(async () => {
    await expect(redirect!(text, 'steer')).resolves.toBe(true)
  })
}

describe('steer mid-turn keeps arrival order (user bubble never above prior output)', () => {
  beforeEach(() => {
    handleEvent = null
    redirect = null
    states = new Map()
    requestGatewayMock.mockClear()
  })

  afterEach(() => {
    cleanup()
    vi.useRealTimers()
    vi.restoreAllMocks()
  })

  it('orders pre-steer output → steer → post-steer output → settled reply', async () => {
    await mountHarness()

    emit({ payload: {}, session_id: SID, type: 'message.start' })
    emit({ payload: { text: 'first half of the answer' }, session_id: SID, type: 'message.delta' })
    await flushDeltas()

    // Mid-turn tool activity belongs to the pre-steer bubble.
    emit({
      payload: { args: { command: 'true' }, name: 'terminal', tool_id: 't1' },
      session_id: SID,
      type: 'tool.start'
    })
    emit({ payload: { name: 'terminal', result: 'ok', tool_id: 't1' }, session_id: SID, type: 'tool.complete' })

    await steer('actually do it differently')

    // Post-steer deltas must seed a FRESH bubble below the correction.
    emit({ payload: { text: 'rebuilt answer after the steer' }, session_id: SID, type: 'message.delta' })
    await flushDeltas()

    const midTurn = states.get(SID)!.messages
    const steerIndex = midTurn.findIndex(message => message.role === 'user')
    const preSteer = midTurn.slice(0, steerIndex)
    const postSteer = midTurn.slice(steerIndex + 1)

    expect(steerIndex, `steer bubble missing: ${transcript().join(' | ')}`).toBeGreaterThan(0)
    // Everything the user had already watched arrive stays ABOVE the bubble…
    expect(preSteer.some(message => chatMessageText(message).includes('first half'))).toBe(true)
    expect(preSteer.every(message => message.role === 'assistant')).toBe(true)
    // …sealed (not pending), so no thinking indicator strands above the steer.
    expect(preSteer.every(message => message.pending !== true)).toBe(true)
    // Post-redirect output continues BELOW the correction, in its own bubble.
    expect(postSteer.length).toBeGreaterThan(0)
    expect(postSteer.some(message => chatMessageText(message).includes('rebuilt answer'))).toBe(true)

    // Completion settles the post-steer bubble in place — order unchanged.
    emit({
      payload: { text: 'rebuilt answer after the steer — done' },
      session_id: SID,
      type: 'message.complete'
    })

    const settled = states.get(SID)!.messages
    const settledSteerIndex = settled.findIndex(message => message.role === 'user')
    const tail = settled.at(-1)

    expect(settledSteerIndex).toBe(steerIndex)
    expect(tail?.role).toBe('assistant')
    expect(chatMessageText(tail!)).toContain('rebuilt answer after the steer — done')
    expect(settled.every(message => message.pending !== true)).toBe(true)
    // The final reply is BELOW the steer bubble, not merged into a row above it.
    expect(settled.indexOf(tail!)).toBeGreaterThan(settledSteerIndex)
  })

  it('steer with no post-steer deltas: completion settles above, bubble stays at the tail', async () => {
    await mountHarness()

    emit({ payload: {}, session_id: SID, type: 'message.start' })
    emit({ payload: { text: 'the whole reply already streamed' }, session_id: SID, type: 'message.delta' })
    await flushDeltas()

    // Steer accepted during the final API call — the reply was already
    // complete, so the correction becomes the NEXT turn's prompt.
    await steer('one more thing')

    emit({ payload: { text: 'the whole reply already streamed' }, session_id: SID, type: 'message.complete' })

    const messages = states.get(SID)!.messages
    const steerIndex = messages.findIndex(message => message.role === 'user')

    // The already-streamed reply settles onto its sealed bubble ABOVE the
    // correction; the correction stays the tail, waiting for its own turn —
    // no duplicate reply row appended below it. Load-bearing assumption: a
    // completion settles the sealed pre-steer bubble in place and never
    // appends/merges below the correction — if the reducer ever changes that,
    // this test is the tripwire.
    expect(steerIndex).toBe(messages.length - 1)
    expect(messages.filter(message => chatMessageText(message).includes('whole reply already streamed'))).toHaveLength(
      1
    )
    expect(messages.every(message => message.pending !== true)).toBe(true)
  })

  it('a second steer in the same turn stays below the first (contiguous run, both below prior output)', async () => {
    await mountHarness()

    emit({ payload: {}, session_id: SID, type: 'message.start' })
    emit({ payload: { text: 'output before any steer' }, session_id: SID, type: 'message.delta' })
    await flushDeltas()

    await steer('first correction')

    emit({ payload: { text: 'output after first steer' }, session_id: SID, type: 'message.delta' })
    await flushDeltas()

    await steer('second correction')

    emit({ payload: { text: 'final output' }, session_id: SID, type: 'message.delta' })
    await flushDeltas()
    emit({ payload: { text: 'final output' }, session_id: SID, type: 'message.complete' })

    const roles = states.get(SID)!.messages.map(message => `${message.role}:${chatMessageText(message).slice(0, 24)}`)

    expect(roles, roles.join(' | ')).toEqual([
      'assistant:output before any steer',
      'user:first correction',
      'assistant:output after first steer',
      'user:second correction',
      'assistant:final output'
    ])
  })

  it.each([false, true])('an accepted steer marks its arrival: the tool keeps running above, the answer streams below (tile=%s)', async tile => {
    await mountHarness(tile)

    emit({ payload: {}, session_id: SID, type: 'message.start' })
    emit({ payload: { text: 'checking first' }, session_id: SID, type: 'message.delta' })
    await flushDeltas()
    emit({ payload: { args: { command: 'sleep 5' }, name: 'terminal', tool_id: 't1' }, session_id: SID, type: 'tool.start' })

    requestGatewayMock.mockImplementationOnce(async () => ({ status: 'queued' }))
    await act(async () => {
      await expect(redirect!('use the other file', 'steer')).resolves.toBe(true)
    })
    expect(requestGatewayMock).toHaveBeenLastCalledWith('session.steer', { session_id: SID, text: 'use the other file' })

    // The steer interrupts nothing: the tool it waits on still reads as running.
    const sealed = states.get(SID)!.messages[0]
    expect(sealed.parts.find(part => part.type === 'tool-call')).not.toHaveProperty('completedAt')
    expect(sealed.pending).toBe(true)

    emit({ payload: { name: 'terminal', result: 'ok', tool_id: 't1' }, session_id: SID, type: 'tool.complete' })
    expect(states.get(SID)!.messages[0].pending).toBe(false)

    emit({ payload: { text: 'switched to the other file' }, session_id: SID, type: 'message.delta' })
    await flushDeltas()
    emit({ payload: { text: 'switched to the other file' }, session_id: SID, type: 'message.complete' })

    expect(transcript()).toEqual([
      'assistant:checking first',
      'user:use the other file',
      'assistant:switched to the other file'
    ])
  })

  it.each([false, true])('an accepted steer paints text still buffered above it, and a lost tool.complete cannot outlive the turn (tile=%s)', async tile => {
    await mountHarness(tile)

    emit({ payload: {}, session_id: SID, type: 'message.start' })
    emit({ payload: { text: 'Hello' }, session_id: SID, type: 'message.delta' })
    await flushDeltas()
    emit({ payload: { args: { command: 'sleep 5' }, name: 'terminal', tool_id: 't1' }, session_id: SID, type: 'tool.start' })
    // Buffered, not yet painted: the delta timer has not fired.
    emit({ payload: { text: ' world' }, session_id: SID, type: 'message.delta' })

    requestGatewayMock.mockImplementationOnce(async () => ({ status: 'queued' }))
    await act(async () => {
      await expect(redirect!('change course', 'steer')).resolves.toBe(true)
    })

    // tool.complete for t1 is lost; the answer to the steer completes the turn.
    emit({ payload: { text: 'new direction' }, session_id: SID, type: 'message.delta' })
    await flushDeltas()
    emit({ payload: { text: 'new direction' }, session_id: SID, type: 'message.complete' })

    expect(transcript()).toEqual(['assistant:Hello world', 'user:change course', 'assistant:new direction'])
    expect(states.get(SID)!.messages.every(message => !message.pending)).toBe(true)
    expect(states.get(SID)!.messages[0].parts.find(part => part.type === 'tool-call')).toHaveProperty('completedAt')
  })

  it.each([false, true])('a refused redirect preserves the continuing stream (tile=%s)', async tile => {
    await mountHarness(tile)

    for (const refusal of ['not_running', 'error'] as const) {
      emit({ payload: {}, session_id: SID, type: 'message.start' })
      emit({ payload: { text: 'before' }, session_id: SID, type: 'message.delta' })
      await flushDeltas()
      const originalId = states.get(SID)!.streamId
      const priorCount = states.get(SID)!.messages.length
      let refuse!: () => void
      requestGatewayMock.mockImplementationOnce(
        () =>
          new Promise((resolve, reject) => {
            refuse = () =>
              refusal === 'error' ? reject(new Error('correction refused')) : resolve({ status: refusal })
          })
      )
      let pending!: Promise<unknown>
      act(() => {
        pending = redirect!('not accepted').catch(error => error)
      })
      emit({ payload: { text: ' during' }, session_id: SID, type: 'message.delta' })
      await flushDeltas()
      await act(async () => {
        refuse()
        const outcome = await pending

        if (refusal === 'error') {expect(outcome).toBeInstanceOf(Error)}
        else {expect(outcome).toBe(false)}
      })
      emit({ payload: { text: ' after' }, session_id: SID, type: 'message.delta' })
      await flushDeltas()
      const current = states.get(SID)!
      expect(current.messages).toHaveLength(priorCount)
      expect(current.streamId).toBe(originalId)
      expect(current.interimBoundaryPending).toBe(false)
      expect(current.messages.at(-1)).toMatchObject({ id: originalId, pending: true })
      expect(chatMessageText(current.messages.at(-1)!)).toBe('before during after')
      emit({ payload: { text: 'before during after done' }, session_id: SID, type: 'message.complete' })
      expect(states.get(SID)!.messages).toHaveLength(priorCount)
      expect(chatMessageText(states.get(SID)!.messages.at(-1)!)).toBe('before during after done')
    }
  })

  it.each([false, true])('a delayed accepted redirect does not move its correction past a new reply (tile=%s)', async tile => {
    await mountHarness(tile)
    emit({ payload: {}, session_id: SID, type: 'message.start' })
    emit({ payload: { text: 'old reply' }, session_id: SID, type: 'message.delta' })
    await flushDeltas()
    let accept!: () => void
    requestGatewayMock.mockImplementationOnce(() => new Promise(resolve => { accept = () => resolve({ status: 'redirected' }) }))
    let pending!: Promise<unknown>
    act(() => { pending = redirect!('correction') })
    emit({ payload: { text: 'old reply' }, session_id: SID, type: 'message.complete' })
    emit({ payload: {}, session_id: SID, type: 'message.start' })
    emit({ payload: { text: 'new reply' }, session_id: SID, type: 'message.delta' })
    await flushDeltas()
    const newStreamId = states.get(SID)!.streamId
    await act(async () => { accept(); await pending })
    expect(states.get(SID)!.streamId).toBe(newStreamId)
    expect(states.get(SID)!.messages.map(message => chatMessageText(message))).toEqual(['old reply', 'correction', 'new reply'])
  })

  it('a redirect the gateway rejects discards the optimistic bubble instead of stranding it', async () => {
    await mountHarness()

    emit({ payload: {}, session_id: SID, type: 'message.start' })
    emit({ payload: { text: 'streaming along' }, session_id: SID, type: 'message.delta' })
    await flushDeltas()

    requestGatewayMock.mockImplementationOnce(async () => ({ status: 'not_running' }))

    await act(async () => {
      await expect(redirect!('too late')).resolves.toBe(false)
    })

    // The rejected correction never reached the model — a bubble for it would
    // lie about the transcript. No user row, stream still live below.
    expect(states.get(SID)!.messages.some(message => message.role === 'user')).toBe(false)

    emit({ payload: { text: 'streaming along — done' }, session_id: SID, type: 'message.complete' })
    expect(chatMessageText(states.get(SID)!.messages.at(-1)!)).toContain('done')
  })

  it('a redirect inside a regenerated turn keeps the partial reply and the correction on screen', async () => {
    await mountHarness()

    // Regenerate armed the turn: the old reply is a hidden branch variant and
    // the live stream joins its branch group.
    act(() => {
      states.set(SID, {
        ...createClientSessionState(),
        busy: true,
        messages: [
          { id: 'u1', role: 'user', parts: [{ type: 'text', text: 'the question' }] },
          {
            id: 'a-old',
            role: 'assistant',
            parts: [{ type: 'text', text: 'old answer' }],
            branchGroupId: 'g1',
            hidden: true
          }
        ],
        pendingBranchGroup: 'g1'
      })
    })

    // What the chat surface renders: the store projected through the runtime
    // repository and the incremental external-store runtime.
    const repository = renderHook(
      ({ messages }: { messages: ChatMessage[] }) => useRuntimeMessageRepository(messages),
      {
        initialProps: { messages: [] as ChatMessage[] }
      }
    )

    const runtime = new IncrementalExternalStoreRuntimeCore({ messages: [], onNew: async () => undefined })

    const onScreen = () => {
      const state = states.get(SID)!

      repository.rerender({ messages: state.messages })
      runtime.setAdapter({
        isRunning: state.busy,
        messageRepository: repository.result.current,
        onNew: async () => undefined
      })

      return runtime.threads
        .getMainThreadRuntimeCore()
        .messages.map(
          message => `${message.role}:${message.content.map(part => ('text' in part ? part.text : '')).join('')}`
        )
        .filter(row => row !== 'assistant:')
    }

    emit({ payload: {}, session_id: SID, type: 'message.start' })
    emit({ payload: { text: 'partial regenerated reply' }, session_id: SID, type: 'message.delta' })
    await flushDeltas()
    expect(onScreen()).toEqual(['user:the question', 'assistant:partial regenerated reply'])

    await steer('actually, shorter')
    emit({ payload: { text: 'short answer' }, session_id: SID, type: 'message.delta' })
    await flushDeltas()

    const expected = [
      'user:the question',
      'assistant:partial regenerated reply',
      'user:actually, shorter',
      'assistant:short answer'
    ]

    expect(onScreen()).toEqual(expected)

    emit({ payload: { text: 'short answer' }, session_id: SID, type: 'message.complete' })
    expect(onScreen()).toEqual(expected)
  })
})

// A steer joins the turn at its next tool boundary, where the agent appends it
// after the batch's results. The model keeps thinking and writing the reply it
// is on until then, so that reply stays one bubble ABOVE the steer, and only
// output after `steer.delivered` streams below it. Gateways that announce the
// event say so in the steer's ACK (`delivery_event`); older ones keep the
// seal-at-arrival behaviour above.
describe('a steer waits under the live reply until the model reads it', () => {
  beforeEach(() => {
    handleEvent = null
    redirect = null
    states = new Map()
    requestGatewayMock.mockClear()
  })

  afterEach(() => {
    cleanup()
    vi.useRealTimers()
    vi.restoreAllMocks()
  })

  /** Start a reasoning turn whose first thinking chunk is on screen. */
  const thinkingTurn = async (tile = false) => {
    await mountHarness(tile)
    emit({ payload: {}, session_id: SID, type: 'message.start' })
    emit({ payload: { text: 'weighing option A' }, session_id: SID, type: 'reasoning.delta' })
    await flushDeltas()
  }

  /** The rest of the thinking, the tool it leads to, and its result. */
  const finishThinkingWithTool = async () => {
    emit({ payload: { text: ', then B' }, session_id: SID, type: 'reasoning.delta' })
    await flushDeltas()
    emit({ payload: { args: { command: 'ls' }, name: 'terminal', tool_id: 't1' }, session_id: SID, type: 'tool.start' })
    emit({ payload: { name: 'terminal', result: 'ok', tool_id: 't1' }, session_id: SID, type: 'tool.complete' })
  }

  const answer = async (text: string) => {
    emit({ payload: { text }, session_id: SID, type: 'message.delta' })
    await flushDeltas()
    emit({ payload: { text }, session_id: SID, type: 'message.complete' })
  }

  const settledCleanly = () => {
    const state = states.get(SID)!

    expect(state.messages.every(message => !message.pending && !message.steerPending)).toBe(true)
  }

  it.each([false, true])('thinking that continues after the steer stays one block above it (tile=%s)', async tile => {
    await thinkingTurn(tile)
    await acceptSteer('prefer option B')
    await finishThinkingWithTool()

    expect(layout()).toEqual(['assistant:think(weighing option A, then B) + tool(t1)', 'user:prefer option B'])

    emit({ payload: delivery(['prefer option B']), session_id: SID, type: 'steer.delivered' })
    await answer('going with B')

    expect(layout()).toEqual([
      'assistant:think(weighing option A, then B) + tool(t1)',
      'user:prefer option B',
      'assistant:going with B'
    ])
    settledCleanly()
  })

  it.each([false, true])('delivery that beats the ACK still seals above the steer (tile=%s)', async tile => {
    await thinkingTurn(tile)
    let ack!: () => void
    requestGatewayMock.mockImplementationOnce(
      () => new Promise(resolve => (ack = () => resolve({ status: 'queued', delivery_event: true })))
    )
    let pending!: Promise<boolean>
    act(() => {
      pending = redirect!('prefer option B', 'steer')
    })

    await finishThinkingWithTool()
    emit({ payload: delivery(['prefer option B']), session_id: SID, type: 'steer.delivered' })
    await act(async () => {
      ack()
      await expect(pending).resolves.toBe(true)
    })
    await answer('going with B')

    expect(layout()).toEqual([
      'assistant:think(weighing option A, then B) + tool(t1)',
      'user:prefer option B',
      'assistant:going with B'
    ])
    settledCleanly()
  })

  it('a delivery releases only the steers it drained; a steer sent after it stays held', async () => {
    await thinkingTurn()
    await acceptSteer('prefer option B')
    // The second steer is sent while the first is being delivered; its ACK comes later.
    let ack!: () => void
    requestGatewayMock.mockImplementationOnce(
      () => new Promise(resolve => (ack = () => resolve({ status: 'queued', delivery_event: true })))
    )
    let pending!: Promise<boolean>
    act(() => {
      pending = redirect!('and keep it short', 'steer')
    })
    await finishThinkingWithTool()
    emit({ payload: delivery(['prefer option B']), session_id: SID, type: 'steer.delivered' })
    await act(async () => {
      ack()
      await expect(pending).resolves.toBe(true)
    })
    // Answers only the first steer; the second is still unread.
    emit({ payload: { text: 'B it is' }, session_id: SID, type: 'reasoning.delta' })
    await flushDeltas()
    emit({ payload: { args: { command: 'ls' }, name: 'terminal', tool_id: 't2' }, session_id: SID, type: 'tool.start' })
    emit({ payload: { name: 'terminal', result: 'ok', tool_id: 't2' }, session_id: SID, type: 'tool.complete' })

    expect(layout()).toEqual([
      'assistant:think(weighing option A, then B) + tool(t1)',
      'user:prefer option B',
      'assistant:think(B it is) + tool(t2)',
      'user:and keep it short'
    ])

    emit({ payload: delivery(['and keep it short']), session_id: SID, type: 'steer.delivered' })
    await answer('B, briefly')

    expect(layout().slice(-2)).toEqual(['user:and keep it short', 'assistant:B, briefly'])
    settledCleanly()
  })

  it('a steer sent before its turn starts stays held through message.start', async () => {
    await mountHarness()
    let ack!: () => void
    requestGatewayMock.mockImplementationOnce(
      () => new Promise(resolve => (ack = () => resolve({ status: 'queued', delivery_event: true })))
    )
    let pending!: Promise<boolean>
    act(() => {
      pending = redirect!('prefer option B', 'steer')
    })
    emit({ payload: {}, session_id: SID, type: 'message.start' })
    emit({ payload: { text: 'weighing option A' }, session_id: SID, type: 'reasoning.delta' })
    await flushDeltas()
    await act(async () => {
      ack()
      await expect(pending).resolves.toBe(true)
    })
    await finishThinkingWithTool()
    emit({ payload: delivery(['prefer option B']), session_id: SID, type: 'steer.delivered' })
    await answer('going with B')

    expect(layout()).toEqual([
      'assistant:think(weighing option A, then B) + tool(t1)',
      'user:prefer option B',
      'assistant:going with B'
    ])
    settledCleanly()
  })

  it('a drain that also carried a steer this window did not send releases only its own', async () => {
    await thinkingTurn()
    await acceptSteer('one')
    await acceptSteer('two')
    await finishThinkingWithTool()
    // A queued prompt's transfer (or another client) was drained with "one"; "two" is still buffered.
    emit({ payload: delivery(['one', 'queued card text']), session_id: SID, type: 'steer.delivered' })
    emit({ payload: { text: 'answering one' }, session_id: SID, type: 'reasoning.delta' })
    await flushDeltas()

    expect(layout()).toEqual([
      'assistant:think(weighing option A, then B) + tool(t1)',
      'user:one',
      'assistant:think(answering one)',
      'user:two'
    ])
    expect(states.get(SID)!.messages.at(-1)?.steerPending).toBe('two')
  })

  it('a delivery of only steers this window did not send seals nothing here', async () => {
    await thinkingTurn()
    await acceptSteer('mine')
    await finishThinkingWithTool()
    emit({ payload: delivery(['from another window']), session_id: SID, type: 'steer.delivered' })
    emit({ payload: { text: ', still going' }, session_id: SID, type: 'reasoning.delta' })
    await flushDeltas()

    // Same bubble, still above the steer: nothing was sealed.
    expect(layout()).toEqual(['assistant:think(weighing option A, then B) + tool(t1) + think(, still going)', 'user:mine'])
  })

  it.each([
    ['NEL', '\u0085prefer B\u0085', 'prefer B'],
    ['BOM', '\ufeffprefer B', '\ufeffprefer B']
  ])('a steer is matched whatever edge whitespace the agent strips (%s)', async (_label, sent, stored) => {
    await thinkingTurn()
    await acceptSteer(sent)
    await finishThinkingWithTool()
    // agent.steer() stores Python's str.strip() of the text, which differs from JS trim().
    emit({ payload: delivery([stored]), session_id: SID, type: 'steer.delivered' })
    await answer('going with B')

    expect(layout().at(-1)).toBe('assistant:going with B')
    settledCleanly()
  })

  it('an older payload without texts releases the oldest count', async () => {
    await thinkingTurn()
    await acceptSteer('one')
    await acceptSteer('two')
    await finishThinkingWithTool()
    emit({ payload: { count: 1 }, session_id: SID, type: 'steer.delivered' })
    emit({ payload: { text: 'answering one' }, session_id: SID, type: 'reasoning.delta' })
    await flushDeltas()

    expect(layout().slice(1)).toEqual(['user:one', 'assistant:think(answering one)', 'user:two'])
  })

  it('an older gateway (no delivery_event) keeps sealing at arrival', async () => {
    await thinkingTurn()
    requestGatewayMock.mockImplementationOnce(async () => ({ status: 'queued' }))
    await act(async () => {
      await expect(redirect!('prefer option B', 'steer')).resolves.toBe(true)
    })
    await finishThinkingWithTool()
    await answer('going with B')

    expect(layout()).toEqual([
      'assistant:think(weighing option A)',
      'user:prefer option B',
      'assistant:think(, then B) + tool(t1) + going with B'
    ])
    settledCleanly()
  })

  it('an interrupting redirect after a held steer answers below both corrections', async () => {
    await thinkingTurn()
    await acceptSteer('prefer option B')
    // Interrupt mode: the gateway cancels the request and redirects the turn.
    await act(async () => {
      await expect(redirect!('stop, use C', 'interrupt')).resolves.toBe(true)
    })
    await answer('using C')

    expect(layout()).toEqual([
      'assistant:think(weighing option A)',
      'user:prefer option B',
      'user:stop, use C',
      'assistant:using C'
    ])
    settledCleanly()
  })

  it('a refused steer leaves no hold behind', async () => {
    await thinkingTurn()
    requestGatewayMock.mockImplementationOnce(async () => ({ status: 'rejected' }))
    await act(async () => {
      await expect(redirect!('too late', 'steer')).resolves.toBe(false)
    })

    expect(states.get(SID)!.messages.some(message => message.steerPending)).toBe(false)
    await answer('done')
    expect(layout()).toEqual(['assistant:think(weighing option A) + done'])
  })

  it('a steer that never meets a tool boundary stays below the finished reply as the next prompt', async () => {
    await mountHarness()
    emit({ payload: {}, session_id: SID, type: 'message.start' })
    emit({ payload: { text: 'short answer' }, session_id: SID, type: 'message.delta' })
    await flushDeltas()
    await acceptSteer('also add a summary')
    emit({ payload: { text: ' with details' }, session_id: SID, type: 'message.delta' })
    await flushDeltas()
    emit({ payload: { text: 'short answer with details' }, session_id: SID, type: 'message.complete' })

    expect(layout()).toEqual(['assistant:short answer with details', 'user:also add a summary'])
    settledCleanly()
  })

  it('a reply row created while the steer is held lands above it', async () => {
    await mountHarness()
    emit({ payload: {}, session_id: SID, type: 'message.start' })
    emit({ payload: { text: 'interim note' }, session_id: SID, type: 'message.delta' })
    await flushDeltas()
    emit({ payload: { text: 'interim note', already_streamed: true }, session_id: SID, type: 'message.interim' })
    await acceptSteer('be brief')
    emit({ payload: { text: 'still the old plan' }, session_id: SID, type: 'reasoning.delta' })
    await flushDeltas()

    expect(layout()).toEqual(['assistant:interim note', 'assistant:think(still the old plan)', 'user:be brief'])
  })
})
