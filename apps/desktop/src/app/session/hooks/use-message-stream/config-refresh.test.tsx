import { QueryClient } from '@tanstack/react-query'
import { act, cleanup, render } from '@testing-library/react'
import { useEffect, useRef } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ClientSessionState } from '@/app/types'
import { createClientSessionState } from '@/lib/chat-runtime'
import type { RpcEvent } from '@/types/hermes'

import { useMessageStream } from './index'

// B4(a): session.info used to trigger an immediate, unconditional
// refreshHermesConfig() (= /api/config + /api/config/defaults) per event.
// The handler must now debounce the refetch and skip events that can't have
// changed config-derived state (background session, no personality/cwd field).

const ACTIVE_SID = 'session-active'

let handleEvent: ((event: RpcEvent) => void) | null = null
let refreshHermesConfig: ReturnType<typeof vi.fn>

function Harness() {
  const activeSessionIdRef = useRef<string | null>(ACTIVE_SID)
  const sessionStateByRuntimeIdRef = useRef(new Map<string, ClientSessionState>())
  const queryClientRef = useRef(new QueryClient())

  const stream = useMessageStream({
    activeSessionIdRef,
    hydrateFromStoredSession: vi.fn(async () => undefined),
    queryClient: queryClientRef.current,
    refreshHermesConfig: refreshHermesConfig as unknown as () => Promise<void>,
    scheduleSessionsRefresh: vi.fn(),
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
  }, [stream.handleGatewayEvent])

  return null
}

const sessionInfo = (sessionId: string, payload: Record<string, unknown> = {}) =>
  act(() => handleEvent!({ payload, session_id: sessionId, type: 'session.info' }))

describe('session.info config-refresh damping', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    handleEvent = null
    refreshHermesConfig = vi.fn(async () => undefined)
    render(<Harness />)
  })

  afterEach(() => {
    cleanup()
    vi.useRealTimers()
  })

  it('debounces a burst of active-session heartbeats into one trailing refetch', async () => {
    sessionInfo(ACTIVE_SID, { running: true })
    sessionInfo(ACTIVE_SID, { running: true })
    sessionInfo(ACTIVE_SID, { running: false })

    expect(refreshHermesConfig).not.toHaveBeenCalled()

    await act(async () => {
      await vi.advanceTimersByTimeAsync(5_000)
    })

    expect(refreshHermesConfig).toHaveBeenCalledTimes(1)
  })

  it('skips background-session events that carry no config-backed fields', async () => {
    sessionInfo('session-background', { running: false })
    sessionInfo('session-background', { usage: { total: 1 } })

    await act(async () => {
      await vi.advanceTimersByTimeAsync(30_000)
    })

    expect(refreshHermesConfig).not.toHaveBeenCalled()
  })

  it('still refetches when a background event carries a config-backed field (personality)', async () => {
    sessionInfo('session-background', { personality: 'hacker' })

    await act(async () => {
      await vi.advanceTimersByTimeAsync(5_000)
    })

    expect(refreshHermesConfig).toHaveBeenCalledTimes(1)
  })

  it('refetches again for events after the debounce window', async () => {
    sessionInfo(ACTIVE_SID, { running: true })
    await act(async () => {
      await vi.advanceTimersByTimeAsync(5_000)
    })
    expect(refreshHermesConfig).toHaveBeenCalledTimes(1)

    sessionInfo(ACTIVE_SID, { running: false })
    await act(async () => {
      await vi.advanceTimersByTimeAsync(5_000)
    })
    expect(refreshHermesConfig).toHaveBeenCalledTimes(2)
  })
})
