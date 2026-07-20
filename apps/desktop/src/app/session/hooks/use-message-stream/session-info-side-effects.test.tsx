import { QueryClient } from '@tanstack/react-query'
import { act, cleanup, render, waitFor } from '@testing-library/react'
import { useEffect, useRef } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ClientSessionState } from '@/app/types'
import { createClientSessionState } from '@/lib/chat-runtime'
import { setCurrentModel, setCurrentProvider } from '@/store/session'
import type { RpcEvent } from '@/types/hermes'

import { useMessageStream } from './index'

// Per-turn REST amplification guards: session.info must not refetch config for
// background sessions nor invalidate the model-options catalog when the model
// string is merely PRESENT (the backend stamps it on every event) rather than
// actually changed. message.complete must coalesce sidebar refreshes.

const ACTIVE_SID = 'session-active'
let handleEvent: ((event: RpcEvent) => void) | null = null
let refreshHermesConfig: ReturnType<typeof vi.fn<() => Promise<void>>>
let refreshSessions: ReturnType<typeof vi.fn<() => Promise<void>>>
let queryClient: QueryClient

function Harness() {
  const activeSessionIdRef = useRef<string | null>(ACTIVE_SID)
  const sessionStateByRuntimeIdRef = useRef(new Map<string, ClientSessionState>())

  const stream = useMessageStream({
    activeSessionIdRef,
    hydrateFromStoredSession: vi.fn(async () => undefined),
    queryClient,
    refreshHermesConfig,
    refreshSessions,
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

async function mountStream() {
  render(<Harness />)
  await waitFor(() => expect(handleEvent).not.toBeNull())
}

const sessionInfo = (sessionId: string, payload: Record<string, unknown>) =>
  act(() => handleEvent!({ payload, session_id: sessionId, type: 'session.info' }))

beforeEach(() => {
  handleEvent = null
  refreshHermesConfig = vi.fn<() => Promise<void>>(async () => undefined)
  refreshSessions = vi.fn<() => Promise<void>>(async () => undefined)
  queryClient = new QueryClient()
  setCurrentModel('')
  setCurrentProvider('')
})

afterEach(() => {
  cleanup()
  setCurrentModel('')
  setCurrentProvider('')
  vi.useRealTimers()
  vi.restoreAllMocks()
})

describe('session.info config refetch gating', () => {
  it('coalesces active-session bursts into one trailing config fetch', async () => {
    // Mount under real timers (waitFor), then freeze time for the debounce.
    await mountStream()
    vi.useFakeTimers()

    sessionInfo(ACTIVE_SID, { model: 'm1', running: true })
    sessionInfo(ACTIVE_SID, { model: 'm1', running: false })
    sessionInfo(ACTIVE_SID, { model: 'm1', title: 't' })

    expect(refreshHermesConfig).not.toHaveBeenCalled()

    await act(async () => {
      await vi.advanceTimersByTimeAsync(400)
    })

    expect(refreshHermesConfig).toHaveBeenCalledTimes(1)
  })

  it('never fetches config for a background session heartbeat', async () => {
    await mountStream()
    vi.useFakeTimers()

    sessionInfo('session-background', { model: 'm1', running: true })
    sessionInfo('session-background', { model: 'm1', running: false })

    await act(async () => {
      await vi.advanceTimersByTimeAsync(400)
    })

    expect(refreshHermesConfig).not.toHaveBeenCalled()
  })
})

describe('session.info model-options invalidation gating', () => {
  it('skips invalidation when model/provider merely restate the known values', async () => {
    await mountStream()
    const invalidate = vi.spyOn(queryClient, 'invalidateQueries')

    // Seed the session's cached runtime state.
    sessionInfo(ACTIVE_SID, { model: 'm1', provider: 'p1', running: true })
    invalidate.mockClear()

    // Turn-end heartbeat restating the same model/provider — the pre-fix path
    // invalidated (and refetched the provider catalog) on every one of these.
    sessionInfo(ACTIVE_SID, { model: 'm1', provider: 'p1', running: false })

    expect(invalidate).not.toHaveBeenCalled()
  })

  it('invalidates when the session model actually changes', async () => {
    await mountStream()
    const invalidate = vi.spyOn(queryClient, 'invalidateQueries')

    sessionInfo(ACTIVE_SID, { model: 'm1', provider: 'p1', running: true })
    invalidate.mockClear()

    sessionInfo(ACTIVE_SID, { model: 'm2', provider: 'p1', running: true })

    expect(invalidate).toHaveBeenCalledWith({ queryKey: ['model-options', ACTIVE_SID] })
  })
})

describe('message.complete sidebar refresh coalescing', () => {
  it('collapses near-simultaneous completions into one refresh', async () => {
    await mountStream()
    vi.useFakeTimers()

    act(() => handleEvent!({ payload: { text: 'a' }, session_id: 's1', type: 'message.complete' }))
    act(() => handleEvent!({ payload: { text: 'b' }, session_id: 's2', type: 'message.complete' }))

    expect(refreshSessions).not.toHaveBeenCalled()

    await act(async () => {
      await vi.advanceTimersByTimeAsync(400)
    })

    expect(refreshSessions).toHaveBeenCalledTimes(1)
  })
})
