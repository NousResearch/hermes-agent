// Repro for "when I steer it often sends out of order — a user bubble way
// above" (#73793 / #83151 class). Drives the REAL stream reducer
// (useMessageStream.handleGatewayEvent) and the REAL mid-turn insert
// (appendMidTurnUserMessage — what redirectPrompt's optimistic append uses)
// through a full steered turn, and asserts transcript ORDER:
//
//   pre-steer output → steer bubble → post-steer output → settled reply
//
// The steer bubble must land AFTER every assistant row that had already
// streamed when it was typed, and every later delta / tool event / completion
// must land BELOW it — never spliced above, never merged into the sealed
// pre-steer bubble.
import { QueryClient } from '@tanstack/react-query'
import { act, cleanup, render } from '@testing-library/react'
import { useEffect, useRef } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { appendMidTurnUserMessage } from '@/app/session/hooks/use-prompt-actions/rewind'
import type { ClientSessionState } from '@/app/types'
import { chatMessageText } from '@/lib/chat-messages'
import { createClientSessionState } from '@/lib/chat-runtime'
import type { RpcEvent } from '@/types/hermes'

import { STREAM_DELTA_FLUSH_MS } from './utils'

import { useMessageStream } from './index'

const SID = 'steer-order-session'

let handleEvent: ((event: RpcEvent) => void) | null = null
let states: Map<string, ClientSessionState>

function Harness() {
  const activeSessionIdRef = useRef<null | string>(SID)
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

      return next
    }
  })

  useEffect(() => {
    handleEvent = stream.handleGatewayEvent
    states = sessionStateByRuntimeIdRef.current
  }, [stream.handleGatewayEvent])

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

/** The optimistic steer insert redirectPrompt performs (appendAfterActiveReply). */
const steer = (text: string) =>
  act(() => {
    const current = states.get(SID) ?? createClientSessionState()

    states.set(
      SID,
      appendMidTurnUserMessage(current, {
        id: `user-${Date.now()}-steer`,
        role: 'user',
        parts: [{ type: 'text', text }]
      })
    )
  })

const transcript = () =>
  (states.get(SID)?.messages ?? []).map(message => `${message.role}:${chatMessageText(message).slice(0, 30)}`)

describe('steer mid-turn keeps arrival order (user bubble never above prior output)', () => {
  beforeEach(() => {
    handleEvent = null
    states = new Map()
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

    steer('actually do it differently')

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
    steer('one more thing')

    emit({ payload: { text: 'the whole reply already streamed' }, session_id: SID, type: 'message.complete' })

    const messages = states.get(SID)!.messages
    const steerIndex = messages.findIndex(message => message.role === 'user')

    // The already-streamed reply settles onto its sealed bubble ABOVE the
    // correction; the correction stays the tail, waiting for its own turn —
    // no duplicate reply row appended below it.
    expect(steerIndex).toBe(messages.length - 1)
    expect(messages.filter(message => chatMessageText(message).includes('whole reply already streamed'))).toHaveLength(1)
    expect(messages.every(message => message.pending !== true)).toBe(true)
  })

  it('a second steer in the same turn stays below the first (contiguous run, both below prior output)', async () => {
    await mountHarness()

    emit({ payload: {}, session_id: SID, type: 'message.start' })
    emit({ payload: { text: 'output before any steer' }, session_id: SID, type: 'message.delta' })
    await flushDeltas()

    steer('first correction')

    emit({ payload: { text: 'output after first steer' }, session_id: SID, type: 'message.delta' })
    await flushDeltas()

    steer('second correction')

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
})
