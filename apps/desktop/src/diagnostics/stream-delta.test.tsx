import { QueryClient } from '@tanstack/react-query'
import { act, cleanup, render } from '@testing-library/react'
import { useEffect, useRef } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { useMessageStream } from '@/app/session/hooks/use-message-stream'
import type { ClientSessionState } from '@/app/types'
import { createClientSessionState } from '@/lib/chat-runtime'
import type { RpcEvent } from '@/types/hermes'

import {
  armDiagnostics,
  disarmDiagnostics,
  type GatewayEventAppliedEvent,
  readDiagnosticsCapture,
  type StreamDeltaAppliedEvent
} from './capture'

// The sanitization proof: this string is the entire content of a streamed
// delta, so if ANY of it reaches the ring buffer the capture leaks message text.
const MARKER = 'PRIVATE_MARKER_9137 the user asked about their salary'

const SID = 'session-1'
const SID2 = 'session-2'

let appendAssistantDelta: ((sessionId: string, delta: string) => void) | null = null
let handleEvent: ((event: RpcEvent) => void) | null = null
let states: Map<string, ClientSessionState>

function Harness() {
  const activeSessionIdRef = useRef<string | null>(SID)
  const sessionStateByRuntimeIdRef = useRef(states)
  const queryClientRef = useRef(new QueryClient())

  const stream = useMessageStream({
    activeSessionIdRef,
    hydrateFromStoredSession: vi.fn(async () => undefined),
    queryClient: queryClientRef.current,
    refreshHermesConfig: vi.fn(async () => undefined),
    refreshSessions: vi.fn(async () => undefined),
    sessionStateByRuntimeIdRef,
    updateSessionState: (sessionId, updater) => {
      const next = updater(states.get(sessionId) ?? createClientSessionState())
      states.set(sessionId, next)

      return next
    }
  })

  useEffect(() => {
    appendAssistantDelta = stream.appendAssistantDelta
    handleEvent = stream.handleGatewayEvent
  }, [stream.appendAssistantDelta, stream.handleGatewayEvent])

  return null
}

const streamDeltaEvents = () =>
  (readDiagnosticsCapture()?.events ?? []).filter(
    (event): event is StreamDeltaAppliedEvent => event.type === 'stream_delta_applied'
  )

const gatewayEventEvents = () =>
  (readDiagnosticsCapture()?.events ?? []).filter(
    (event): event is GatewayEventAppliedEvent => event.type === 'gateway_event_applied'
  )

describe('stream-delta diagnostics on the real flush path', () => {
  let rafCallbacks: FrameRequestCallback[]
  let now: number

  beforeEach(() => {
    vi.useFakeTimers()
    appendAssistantDelta = null
    handleEvent = null
    states = new Map()
    rafCallbacks = []
    now = 1_000
    vi.spyOn(performance, 'now').mockImplementation(() => now)
    vi.spyOn(window, 'requestAnimationFrame').mockImplementation(cb => {
      rafCallbacks.push(cb)

      return rafCallbacks.length
    })
    vi.spyOn(window, 'cancelAnimationFrame').mockImplementation(() => undefined)
  })

  afterEach(() => {
    cleanup()
    disarmDiagnostics()
    vi.useRealTimers()
    vi.restoreAllMocks()
  })

  it('records counts and durations for an applied flush — never the delta text', async () => {
    armDiagnostics('capture-stream-1', 1_700_000_000_000)

    render(<Harness />)
    expect(appendAssistantDelta).not.toBeNull()

    act(() => appendAssistantDelta!(SID, MARKER))
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0)
    })

    const [event] = streamDeltaEvents()
    expect(event).toBeDefined()
    expect(event.sessions).toBe(1)
    expect(event.queuedChars).toBe(MARKER.length)
    expect(event.historyMessages).toBe(1)
    expect(event.writeMs).toBeGreaterThanOrEqual(0)
    expect(event.path).toBe('timer')
    expect(event.busySessions).toBe(0)

    // The deferred commit frame lands 40ms later and runs for 60ms; both come
    // from the flush path's EXISTING measurement, filled into the same event.
    now = 1_100
    act(() => rafCallbacks[0](1_040))

    expect(event.rafGapMs).toBe(40)
    expect(event.commitMs).toBe(60)
    expect(streamDeltaEvents()).toHaveLength(1)

    // The proof: nothing the user typed or the model said is in the buffer.
    expect(JSON.stringify(readDiagnosticsCapture())).not.toContain('PRIVATE_MARKER_9137')
    expect(JSON.stringify(readDiagnosticsCapture())).not.toContain('salary')
  })

  it('attributes a second busy thread: eager tool-boundary drains and busySessions', async () => {
    armDiagnostics('capture-threads', 1_700_000_000_000)

    render(<Harness />)
    expect(handleEvent).not.toBeNull()

    // Two threads with turns in flight — the second is the one the old
    // instrumentation could not see (its text drains on tool boundaries,
    // outside the timer flush, so `sessions` always read 1).
    act(() => handleEvent!({ payload: {}, session_id: SID, type: 'message.start' }))
    act(() => handleEvent!({ payload: {}, session_id: SID2, type: 'message.start' }))

    act(() => appendAssistantDelta!(SID, MARKER))
    act(() => appendAssistantDelta!(SID2, MARKER))

    // Thread 2 hits a tool call: its queued text flushes eagerly, right now,
    // and leaves the queue before the timer flush can count it.
    act(() => handleEvent!({ payload: { name: 'terminal' }, session_id: SID2, type: 'tool.start' }))

    const [eagerEvent] = streamDeltaEvents()
    expect(eagerEvent).toMatchObject({ busySessions: 2, path: 'eager', queuedChars: MARKER.length, sessions: 1 })
    // Eager drains skip the commit-measurement frame by design.
    expect(eagerEvent.commitMs).toBe(0)

    // The timer flush then only sees thread 1 — but busySessions still says 2.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0)
    })

    const [, timerEvent] = streamDeltaEvents()
    expect(timerEvent).toMatchObject({ busySessions: 2, path: 'timer', queuedChars: MARKER.length, sessions: 1 })

    // Sanitization holds on the new path too.
    expect(JSON.stringify(readDiagnosticsCapture())).not.toContain('PRIVATE_MARKER_9137')
  })

  it('records costly gateway-event dispatches with their type tag', async () => {
    armDiagnostics('capture-dispatch', 1_700_000_000_000)

    render(<Harness />)
    expect(handleEvent).not.toBeNull()

    // Advance the mocked clock on every read: the wrapper's start/end reads
    // alone make any dispatch look 5ms+, which clears the 4ms recording floor
    // deterministically — no reliance on real handler cost in jsdom.
    vi.spyOn(performance, 'now').mockImplementation(() => {
      now += 5

      return now
    })

    act(() => handleEvent!({ payload: {}, session_id: SID, type: 'thinking.delta' }))

    const [event] = gatewayEventEvents()
    expect(event).toBeDefined()
    expect(event.eventType).toBe('thinking.delta')
    expect(event.durationMs).toBeGreaterThanOrEqual(4)
    expect(event.busySessions).toBe(0)
  })

  it('does no per-delta recording work while disarmed', async () => {
    render(<Harness />)

    act(() => appendAssistantDelta!(SID, MARKER))
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0)
    })

    now = 1_100
    act(() => rafCallbacks[0](1_040))

    // The delta still applied normally...
    const part = states.get(SID)?.messages.at(-1)?.parts.at(-1)
    expect(part?.type === 'text' ? part.text : '').toBe(MARKER)
    // ...and the capture side did nothing at all.
    expect(readDiagnosticsCapture()).toBeNull()
  })
})
