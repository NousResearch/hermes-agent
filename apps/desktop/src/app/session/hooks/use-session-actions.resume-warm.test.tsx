import { cleanup, render, waitFor } from '@testing-library/react'
import type { MutableRefObject } from 'react'
import { useEffect } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { $sessions, setSessions } from '@/store/session'

import type { ClientSessionState, SessionInfo } from '../../types'

import { useSessionActions } from './use-session-actions'

// Keep the gateway-profile swap a no-op so resumeSession's awaits settle
// without a real backend.
vi.mock('@/store/profile', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  ensureGatewayProfile: vi.fn(async () => {})
}))

const ref = <T,>(value: T): MutableRefObject<T> => ({ current: value })

function state(storedSessionId: string): ClientSessionState {
  return {
    storedSessionId,
    messages: [],
    branch: '',
    cwd: '',
    model: '',
    provider: '',
    reasoningEffort: '',
    serviceTier: '',
    fast: false,
    yolo: false,
    personality: '',
    busy: false,
    awaitingResponse: false,
    streamId: null,
    sawAssistantPayload: false,
    pendingBranchGroup: null,
    interrupted: false,
    needsInput: false,
    turnStartedAt: null
  }
}

function Harness({
  deps,
  onReady
}: {
  deps: Parameters<typeof useSessionActions>[0]
  onReady: (resume: (storedSessionId: string, replaceRoute?: boolean) => Promise<void>) => void
}) {
  const actions = useSessionActions(deps)

  useEffect(() => {
    onReady(actions.resumeSession)
  }, [actions.resumeSession, onReady])

  return null
}

describe('resumeSession warm-cached switch (#45738)', () => {
  afterEach(() => {
    cleanup()
    setSessions(() => [])
    vi.restoreAllMocks()
  })

  it('flips the foreground view to the warm session synchronously, before the resolve/gateway awaits', async () => {
    const storedA = 'stored-A'
    const storedB = 'stored-B'
    const rtA = 'rt-A'
    const rtB = 'rt-B'

    const activeSessionIdRef = ref<string | null>(rtA)
    const selectedStoredSessionIdRef = ref<string | null>(storedA)
    const runtimeIdByStoredSessionIdRef = ref(new Map<string, string>([[storedA, rtA], [storedB, rtB]]))
    const sessionStateByRuntimeIdRef = ref(new Map<string, ClientSessionState>([[rtA, state(storedA)], [rtB, state(storedB)]]))
    const syncSessionStateToView = vi.fn()

    // Both sessions cached so resolveStoredSession is a pure store hit (no
    // gateway round-trip).
    setSessions(() => [{ id: storedA } as SessionInfo, { id: storedB } as SessionInfo])

    const requestGateway = vi.fn(async () => ({}) as never)

    const deps: Parameters<typeof useSessionActions>[0] = {
      activeSessionId: rtA,
      activeSessionIdRef,
      busyRef: ref(false),
      creatingSessionRef: ref(false),
      ensureSessionState: () => state(storedB),
      getRouteToken: () => 'token',
      navigate: vi.fn() as never,
      requestGateway,
      runtimeIdByStoredSessionIdRef,
      selectedStoredSessionId: storedA,
      selectedStoredSessionIdRef,
      sessionStateByRuntimeIdRef,
      syncSessionStateToView,
      updateSessionState: () => state(storedB)
    }

    let resume: ((s: string, r?: boolean) => Promise<void>) | null = null
    render(<Harness deps={deps} onReady={r => (resume = r)} />)
    await waitFor(() => expect(resume).not.toBeNull())

    // Call without awaiting: the async body runs synchronously up to its first
    // await (resolveStoredSession). The fix repaints the foreground view here;
    // on main the warm repaint only happens AFTER the awaits, so the previous
    // session would still be active at this point.
    const pending = resume!(storedB)

    expect(activeSessionIdRef.current).toBe(rtB)
    expect(syncSessionStateToView).toHaveBeenCalledWith(rtB, sessionStateByRuntimeIdRef.current.get(rtB))

    await pending.catch(() => {})
  })
})
