import { cleanup, render, waitFor } from '@testing-library/react'
import { useEffect } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { createSplitPaneView, type PaneSessionView } from '@/app/chat/pane-view'
import type { SessionInfo } from '@/hermes'
import { $activeSessionId, setSessions } from '@/store/session'
import { $activePaneId } from '@/store/split'

import type { ClientSessionState } from '../../types'

import { useSessionActions } from './use-session-actions'

// Inactive-pane profile pinning (design §4 row 3): the split pane's boot
// restore resumes while the MAIN pane is focused, so it must open its
// profile's socket WITHOUT stealing the active gateway swap — and the resume
// RPC must carry the session's profile explicitly. Once the pane is ACTIVE,
// resumes swap the gateway exactly like selecting a session does today.

const ensureGatewayProfile = vi.fn<(profile: string | null | undefined) => Promise<undefined>>(async () => undefined)

const ensureProfileSocketOpen = vi.fn<(profile: string | null | undefined) => Promise<never>>(
  async () => ({}) as never
)

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  getSessionMessages: vi.fn(async () => ({ messages: [] })),
  setApiRequestProfile: vi.fn()
}))

vi.mock('@/store/profile', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  ensureGatewayProfile: (profile: string | null | undefined) => ensureGatewayProfile(profile)
}))

vi.mock('@/store/gateway', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  ensureProfileSocketOpen: (profile: string | null | undefined) => ensureProfileSocketOpen(profile)
}))

function storedSession(overrides: Partial<SessionInfo> = {}): SessionInfo {
  return {
    ended_at: null,
    id: 'stored-split',
    input_tokens: 0,
    is_active: false,
    last_active: 1,
    message_count: 0,
    model: null,
    output_tokens: 0,
    preview: null,
    source: 'desktop',
    started_at: 1,
    title: 'split session',
    tool_call_count: 0,
    ...overrides
  }
}

function Harness({
  onReady,
  requestGateway,
  view
}: {
  onReady: (resume: (storedSessionId: string) => Promise<void>) => void
  requestGateway: <T>(method: string, params?: Record<string, unknown>) => Promise<T>
  view: PaneSessionView
}) {
  const actions = useSessionActions({
    activeSessionId: null,
    activeSessionIdRef: view.activeSessionIdRef,
    busyRef: view.busyRef,
    creatingSessionRef: view.creatingSessionRef,
    ensureSessionState: () => ({}) as ClientSessionState,
    getRouteToken: () => 'split:token',
    navigate: vi.fn() as never,
    requestGateway,
    runtimeIdByStoredSessionIdRef: { current: new Map<string, string>() },
    selectedStoredSessionId: null,
    selectedStoredSessionIdRef: view.selectedStoredSessionIdRef,
    sessionStateByRuntimeIdRef: { current: new Map<string, ClientSessionState>() },
    syncSessionStateToView: vi.fn(),
    updateSessionState: () => ({}) as ClientSessionState,
    view
  })

  useEffect(() => {
    onReady(actions.resumeSession)
  }, [actions.resumeSession, onReady])

  return null
}

async function resumeWith(view: PaneSessionView) {
  const resumeParamsByCall: Record<string, unknown>[] = []

  const requestGateway = vi.fn(async (method: string, params?: Record<string, unknown>) => {
    if (method === 'session.resume') {
      resumeParamsByCall.push(params ?? {})

      return { info: {}, messages: [], session_id: 'rt-split-1' } as never
    }

    return {} as never
  })

  let resume: ((storedSessionId: string) => Promise<void>) | null = null
  render(<Harness onReady={r => (resume = r)} requestGateway={requestGateway} view={view} />)
  await waitFor(() => expect(resume).not.toBeNull())
  await resume!('stored-split')

  return resumeParamsByCall
}

describe('split-pane resume profile routing', () => {
  afterEach(() => {
    cleanup()
    ensureGatewayProfile.mockClear()
    ensureProfileSocketOpen.mockClear()
    setSessions([])
    $activePaneId.set('main')
    $activeSessionId.set(null)
    vi.restoreAllMocks()
  })

  it('INACTIVE split resume opens the profile socket, never the active swap, and pins the RPC profile', async () => {
    setSessions([storedSession({ profile: 'coder' })])
    $activePaneId.set('main')
    const view = createSplitPaneView()

    const resumeParams = await resumeWith(view)

    // Background restore: socket opened for the session's profile…
    expect(ensureProfileSocketOpen).toHaveBeenCalledWith('coder')
    // …without moving the active gateway out from under the main pane.
    expect(ensureGatewayProfile).not.toHaveBeenCalled()

    // session.resume carries the profile explicitly (socket routing alone
    // isn't trusted for resumes — design §4 payload rules).
    expect(resumeParams).toHaveLength(1)
    expect(resumeParams[0]).toMatchObject({ profile: 'coder', session_id: 'stored-split' })

    // The resumed runtime id landed in the PANE's bundle, not the globals.
    expect(view.$activeSessionId.get()).toBe('rt-split-1')
    expect($activeSessionId.get()).toBeNull()
  })

  it('ACTIVE split resume swaps the live gateway — identical to selecting a session today', async () => {
    setSessions([storedSession({ profile: 'coder' })])
    $activePaneId.set('split')
    const view = createSplitPaneView()

    await resumeWith(view)

    expect(ensureGatewayProfile).toHaveBeenCalledWith('coder')
    expect(ensureProfileSocketOpen).not.toHaveBeenCalled()
  })
})
