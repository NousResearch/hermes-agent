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
import { act, cleanup, render } from '@testing-library/react'
import { useEffect, useRef } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { usePromptActions } from '@/app/session/hooks/use-prompt-actions'
import type { ClientSessionState } from '@/app/types'
import { chatMessageText } from '@/lib/chat-messages'
import { createClientSessionState } from '@/lib/chat-runtime'
import type { RpcEvent } from '@/types/hermes'

import { STREAM_DELTA_FLUSH_MS } from './utils'

import { useMessageStream } from './index'

const SID = 'steer-order-session'

let handleEvent: ((event: RpcEvent) => void) | null = null
let redirect: ((text: string) => Promise<boolean>) | null = null
let states: Map<string, ClientSessionState>

/** The gateway accepts every redirect — these suites pin CLIENT ordering. */
const requestGatewayMock = vi.fn(async (method: string): Promise<unknown> =>
  method === 'session.redirect' ? { status: 'redirected' } : {}
)

const requestGateway = requestGatewayMock as unknown as <T>(
  method: string,
  params?: Record<string, unknown>,
  timeoutMs?: number
) => Promise<T>

function Harness() {
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

  useEffect(() => {
    handleEvent = stream.handleGatewayEvent
    redirect = actions.redirectPrompt
    states = sessionStateByRuntimeIdRef.current
  }, [stream.handleGatewayEvent, actions.redirectPrompt])

  return null
}

async function mountHarness() {
  vi.useFakeTimers()
  render(<Harness />)
  await act(async () => {
    await Promise.resolve()
  })
}

const flushDeltas = async () => {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(STREAM_DELTA_FLUSH_MS)
  })
}

const emit = (event: RpcEvent) => act(() => handleEvent?.(event))

/** A real steer: redirectPrompt's optimistic insert + the gateway round-trip. */
const steer = async (text: string) => {
  await act(async () => {
    await expect(redirect!(text)).resolves.toBe(true)
  })
}

const transcript = () =>
  (states.get(SID)?.messages ?? []).map(message => `${message.role}:${chatMessageText(message).slice(0, 30)}`)

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
    expect(messages.filter(message => chatMessageText(message).includes('whole reply already streamed'))).toHaveLength(1)
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
})
