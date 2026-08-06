import { QueryClient } from '@tanstack/react-query'
import { act, cleanup, render } from '@testing-library/react'
import { useEffect, useRef } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ClientSessionState } from '@/app/types'
import { createClientSessionState } from '@/lib/chat-runtime'
import type { RpcEvent } from '@/types/hermes'

import { STREAM_DELTA_FLUSH_MS } from './utils'

import { useMessageStream } from './index'

const SID = 'stream-session'

let handleEvent: ((event: RpcEvent) => void) | null = null
let states: Map<string, ClientSessionState>

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

      return next
    }
  })

  useEffect(() => {
    handleEvent = stream.handleGatewayEvent
    states = sessionStateByRuntimeIdRef.current
  }, [stream.handleGatewayEvent])

  return null
}

describe('stream delta delivery', () => {
  beforeEach(() => {
    handleEvent = null
    states = new Map()
  })

  afterEach(() => {
    cleanup()
    vi.useRealTimers()
    vi.restoreAllMocks()
  })

  it('delivers a delta while animation frames are parked', async () => {
    // An Electron renderer the compositor has parked: rAF accepts the callback
    // but never runs it. A frame-gated flush strands the answer here until some
    // later focus/input event wakes a frame.
    const rafSpy = vi.spyOn(window, 'requestAnimationFrame').mockImplementation(() => 1)
    vi.useFakeTimers()

    render(<Harness />)
    await act(async () => {
      await Promise.resolve()
    })

    // Reach the frame-gated branch: it is only taken once the coalescing floor
    // has already elapsed since the previous flush. Send one delta, let it
    // flush, then idle past the floor so the NEXT delta schedules immediately.
    act(() => handleEvent?.({ payload: { text: 'first ' }, session_id: SID, type: 'message.delta' }))
    await act(async () => {
      await vi.advanceTimersByTimeAsync(STREAM_DELTA_FLUSH_MS)
    })
    await act(async () => {
      await vi.advanceTimersByTimeAsync(STREAM_DELTA_FLUSH_MS * 2)
    })

    act(() => handleEvent?.({ payload: { text: 'and the rest' }, session_id: SID, type: 'message.delta' }))

    await act(async () => {
      await vi.advanceTimersByTimeAsync(STREAM_DELTA_FLUSH_MS)
    })

    expect(states.get(SID)?.messages.at(-1)?.parts).toEqual([{ type: 'text', text: 'first and the rest' }])
    // The flush must not have depended on a frame: this mock parks every rAF
    // callback, yet the text arrived. runFlush still registers its
    // adaptive-floor measurement callback here; that one is allowed to wait
    // for a frame that may never come.
    expect(rafSpy).toHaveBeenCalled()
  })
})
