import { QueryClient } from '@tanstack/react-query'
import { act, cleanup, render, waitFor } from '@testing-library/react'
import { useEffect, useRef } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { isTargetSessionBusy } from '@/app/session/hooks/use-prompt-actions/utils'
import type { ClientSessionState } from '@/app/types'
import { createClientSessionState } from '@/lib/chat-runtime'
import { modelOptionsQueryKey } from '@/lib/model-options'
import { setCurrentModel, setCurrentProvider } from '@/store/session'
import type { RpcEvent } from '@/types/hermes'

import { useMessageStream } from './index'

// Per-turn REST amplification guards: session.info must not refetch config for
// background sessions nor invalidate the model-options catalog when the model
// string is merely PRESENT (the backend stamps it on every event) rather than
// actually changed. message.complete must coalesce sidebar refreshes.

const ACTIVE_SID = 'session-active'
const ACTIVE_PROFILE = 'compass'
let handleEvent: ((event: RpcEvent) => void) | null = null
let refreshHermesConfig: ReturnType<typeof vi.fn<() => Promise<void>>>
let refreshSessions: ReturnType<typeof vi.fn<() => Promise<void>>>
let hydrateFromStoredSession: ReturnType<typeof vi.fn<() => Promise<void>>>
let queryClient: QueryClient
let sessionStates: Map<string, ClientSessionState> | null = null

function Harness() {
  const activeSessionIdRef = useRef<string | null>(ACTIVE_SID)
  const sessionStateByRuntimeIdRef = useRef(new Map<string, ClientSessionState>())

  sessionStates = sessionStateByRuntimeIdRef.current

  const stream = useMessageStream({
    activeGatewayProfile: ACTIVE_PROFILE,
    activeSessionIdRef,
    hydrateFromStoredSession,
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
  sessionStates = null
  refreshHermesConfig = vi.fn<() => Promise<void>>(async () => undefined)
  refreshSessions = vi.fn<() => Promise<void>>(async () => undefined)
  hydrateFromStoredSession = vi.fn<() => Promise<void>>(async () => undefined)
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

    expect(invalidate).toHaveBeenCalledWith({ queryKey: modelOptionsQueryKey(ACTIVE_PROFILE, ACTIVE_SID) })
  })
})

describe('session.info settles a turn that produced no assistant payload', () => {
  // #46517: a turn that ends without ever emitting an assistant payload (gateway
  // crash mid-stream, provider error before the first delta, agent-build
  // failure) never reaches message.complete, so session.info running=false is
  // the only event that can release it. It used to return state unchanged,
  // latching awaitingResponse/busy until app restart — and because
  // isTargetSessionBusy reads the per-session busy flag as authoritative,
  // submitPrompt and the slash dispatcher then silently refused every send.
  const busyFor = (sessionId: string) => isTargetSessionBusy(Object.fromEntries(sessionStates!), sessionId, false)

  const startTurn = (sessionId: string) =>
    act(() => handleEvent!({ payload: {}, session_id: sessionId, type: 'message.start' }))

  it('leaves the session sendable after a started turn ends with no payload', async () => {
    await mountStream()

    startTurn(ACTIVE_SID)
    expect(busyFor(ACTIVE_SID)).toBe(true)

    sessionInfo(ACTIVE_SID, { running: false })

    const state = sessionStates!.get(ACTIVE_SID)!
    expect(state.awaitingResponse).toBe(false)
    expect(state.busy).toBe(false)
    expect(state.streamId).toBeNull()
    expect(state.turnStartedAt).toBeNull()
    // The predicate submit.ts and slash.ts actually gate on.
    expect(busyFor(ACTIVE_SID)).toBe(false)
  })

  it('keeps waiting when running=false lands before the turn ever started', async () => {
    await mountStream()

    // submit arms busy/awaitingResponse optimistically — and seeds the visible
    // turn clock (turnStartedAt) at Enter — so this heartbeat is the pre-start
    // report, not a finished turn: the spinner must stay up and the send guard
    // must stay closed. turnLive (backend-confirmed) is the discriminator.
    act(() => {
      sessionStates!.set(ACTIVE_SID, {
        ...createClientSessionState(),
        awaitingResponse: true,
        busy: true,
        sawAssistantPayload: false,
        turnStartedAt: Date.now(),
        turnLive: false
      })
    })

    sessionInfo(ACTIVE_SID, { running: false })

    const state = sessionStates!.get(ACTIVE_SID)!
    expect(state.awaitingResponse).toBe(true)
    expect(busyFor(ACTIVE_SID)).toBe(true)
    expect(hydrateFromStoredSession).not.toHaveBeenCalled()
  })

  it('un-latches a background session but does not hydrate its transcript', async () => {
    await mountStream()

    startTurn('session-background')
    sessionInfo('session-background', { running: false })

    // The settle is unscoped — a background session's sidebar dot must clear
    // without the user opening it.
    expect(busyFor('session-background')).toBe(false)
    // The transcript refetch is scoped to the session on screen, so an idle
    // background session does not cost a REST fan-out.
    expect(hydrateFromStoredSession).not.toHaveBeenCalled()
  })

  it('hydrates the foreground transcript once and coalesces the sidebar refresh', async () => {
    await mountStream()

    startTurn(ACTIVE_SID)
    vi.useFakeTimers()

    sessionInfo(ACTIVE_SID, { running: false })
    // Later heartbeats hit the unchanged-state guard, so recovery is edge-only.
    sessionInfo(ACTIVE_SID, { running: false })
    sessionInfo(ACTIVE_SID, { running: false })

    await act(async () => {
      await vi.advanceTimersByTimeAsync(400)
    })

    expect(hydrateFromStoredSession).toHaveBeenCalledTimes(1)
    expect(refreshSessions).toHaveBeenCalledTimes(1)
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
