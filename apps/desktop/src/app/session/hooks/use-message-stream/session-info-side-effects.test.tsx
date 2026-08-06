import { QueryClient } from '@tanstack/react-query'
import { act, cleanup, render, waitFor } from '@testing-library/react'
import { useEffect, useRef } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ClientSessionState } from '@/app/types'
import { createClientSessionState } from '@/lib/chat-runtime'
import { modelOptionsQueryKey } from '@/lib/model-options'

// Only mock @/store/projects — @/store/session is imported for real so
// $currentCwd is a proper nanostores atom with .get() / .set() / .subscribe().
const mocks = vi.hoisted(() => ({
  cwdFollowMock: vi.fn<(_: string) => Promise<void>>(async () => undefined)
}))

vi.mock('@/store/projects', async () => {
  const actual = await vi.importActual('@/store/projects')

  return {
    ...actual,
    followActiveSessionCwd: mocks.cwdFollowMock
  }
})

import { $currentCwd, setCurrentModel, setCurrentProvider } from '@/store/session'
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
let queryClient: QueryClient

function Harness() {
  const activeSessionIdRef = useRef<string | null>(ACTIVE_SID)
  const sessionStateByRuntimeIdRef = useRef(new Map<string, ClientSessionState>())

  const stream = useMessageStream({
    activeGatewayProfile: ACTIVE_PROFILE,
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

    expect(invalidate).toHaveBeenCalledWith({ queryKey: modelOptionsQueryKey(ACTIVE_PROFILE, ACTIVE_SID) })
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

describe('session.info cwd-follow guard', () => {
  beforeEach(() => {
    mocks.cwdFollowMock.mockReset()
    $currentCwd.set('')
  })

  it('does not follow the cwd on the first session.info after reconnect (initial learn)', async () => {
    await mountStream()

    // Initial payload — cwd goes from '' to a real path. Must NOT follow.
    await sessionInfo(ACTIVE_SID, { cwd: '/Users/test/projects/foo' })

    expect(mocks.cwdFollowMock).not.toHaveBeenCalled()
  })

  it('does not follow the cwd on the first session.info after reconnect when the ref is already primed (issue #72491)', async () => {
    // Pre-restart state: the session was active and its cwd already known.
    $currentCwd.set('/Users/test/projects/foo')

    await mountStream()

    // Prime lastCwdInfoSessionRef with a pre-restart session.info for the
    // same session — the ref survives a gateway reconnect.
    await sessionInfo(ACTIVE_SID, { cwd: '/Users/test/projects/foo' })

    // Restart: gateway-bound stores are wiped (cwd resets to '') while the
    // component ref keeps its pre-restart value — the issue #72491 scenario.
    $currentCwd.set('')

    // First session.info after reconnect for the SAME session. This is the
    // initial cwd learn, not a deliberate directory change — must NOT follow.
    await sessionInfo(ACTIVE_SID, { cwd: '/Users/test/projects/foo' })

    expect(mocks.cwdFollowMock).not.toHaveBeenCalled()
  })

  it('follows the cwd on a genuine move (non-empty → different non-empty)', async () => {
    $currentCwd.set('/Users/test/projects/foo')

    await mountStream()

    // Prime lastCwdInfoSessionRef with an initial session.info — on a genuine
    // move the ref was already set by a prior event establishing the session.
    await sessionInfo(ACTIVE_SID, { cwd: '/Users/test/projects/foo' })

    // Genuine move — cwd changes from one non-empty path to another. Must follow.
    await sessionInfo(ACTIVE_SID, { cwd: '/Users/test/projects/bar' })

    expect(mocks.cwdFollowMock).toHaveBeenCalledWith('/Users/test/projects/bar')
  })
})
