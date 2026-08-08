import { QueryClient } from '@tanstack/react-query'
import { act, cleanup, render } from '@testing-library/react'
import { useEffect, useRef } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { useMessageStream } from '@/app/session/hooks/use-message-stream'
import type { ClientSessionState } from '@/app/types'
import { createClientSessionState } from '@/lib/chat-runtime'

import { armDiagnostics, disarmDiagnostics, readDiagnosticsCapture, type StreamDeltaAppliedEvent } from './capture'

// The sanitization proof: this string is the entire content of a streamed
// delta, so if ANY of it reaches the ring buffer the capture leaks message text.
const MARKER = 'PRIVATE_MARKER_9137 the user asked about their salary'

const SID = 'session-1'

let appendAssistantDelta: ((sessionId: string, delta: string) => void) | null = null
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
  }, [stream.appendAssistantDelta])

  return null
}

const streamDeltaEvents = () =>
  (readDiagnosticsCapture()?.events ?? []).filter(
    (event): event is StreamDeltaAppliedEvent => event.type === 'stream_delta_applied'
  )

describe('stream-delta diagnostics on the real flush path', () => {
  let rafCallbacks: FrameRequestCallback[]
  let now: number

  beforeEach(() => {
    vi.useFakeTimers()
    appendAssistantDelta = null
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
