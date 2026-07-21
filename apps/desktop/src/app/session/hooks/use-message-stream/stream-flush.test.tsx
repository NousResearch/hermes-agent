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

describe('useMessageStream delta delivery', () => {
  beforeEach(() => {
    handleEvent = null
    states = new Map()
  })

  afterEach(() => {
    cleanup()
    vi.useRealTimers()
    vi.restoreAllMocks()
  })

  it('flushes queued deltas when animation frames are paused', async () => {
    // Simulate an occluded Electron renderer: rAF accepts work but never runs
    // it. The previous implementation left the delta queued until focus/input.
    vi.spyOn(window, 'requestAnimationFrame').mockImplementation(() => 1)
    vi.useFakeTimers()

    render(<Harness />)
    await act(async () => {
      await Promise.resolve()
    })
    expect(handleEvent).not.toBeNull()

    act(() => handleEvent!({ payload: { text: 'still streaming' }, session_id: SID, type: 'message.delta' }))

    await act(async () => {
      await vi.advanceTimersByTimeAsync(STREAM_DELTA_FLUSH_MS)
    })

    const message = states.get(SID)?.messages.at(-1)
    expect(message?.parts).toEqual([{ type: 'text', text: 'still streaming' }])
  })
})
