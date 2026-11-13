// Branching a chat another profile owns must not let that profile's runtime
// overwrite the ACTIVE profile's approval chip: forkBranch passes the branch's
// owner (the parent row's route) to applyRuntimeInfo, which only reconciles the
// chip when the owner is the active gateway.
import { cleanup, render, waitFor } from '@testing-library/react'
import type { MutableRefObject } from 'react'
import { useEffect } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { type SessionInfo } from '@/hermes'
import { $approvalModes, approvalModeForProfile, reconcileApprovalModeForProfile } from '@/store/approval-mode'
import { requestGatewayForAgent } from '@/store/gateway'
import { $activeGatewayProfile } from '@/store/profile'
import { setMessages, setSelectedStoredSessionId, setSessions } from '@/store/session'
import { $sessionTiles } from '@/store/session-states'

import type { ClientSessionState } from '../../../types'

import { useSessionActions } from './index'

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  deleteSession: vi.fn(),
  getSession: vi.fn(),
  getAllSessionMessages: vi.fn(),
  getLatestSessionMessages: vi.fn(),
  listAllProfileSessions: vi.fn(),
  setApiRequestProfile: vi.fn(),
  setSessionArchived: vi.fn()
}))

vi.mock('@/store/profile', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  ensureGatewayAgent: vi.fn().mockResolvedValue(undefined),
  ensureGatewayProfile: vi.fn().mockResolvedValue(undefined)
}))

vi.mock('@/store/gateway', async importOriginal => {
  const original = await importOriginal<Record<string, unknown>>()

  return {
    ...original,
    // Default-preserving spy: tests that route by the active source override it.
    activeGatewayConnectionId: vi.fn(original.activeGatewayConnectionId as () => null | string),
    requestGatewayForAgent: vi.fn(),
    requestGatewayForProfile: vi.fn(),
    retainGatewayForAgent: vi.fn(async () => () => undefined)
  }
})

vi.mock('@/components/pane-shell/tree/store', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  noteActiveTreeGroup: vi.fn(),
  revealTreePane: vi.fn()
}))

function storedSession(overrides: Partial<SessionInfo> = {}): SessionInfo {
  return {
    ended_at: null,
    id: 'stored-1',
    input_tokens: 0,
    is_active: false,
    last_active: 1,
    message_count: 0,
    model: null,
    output_tokens: 0,
    preview: null,
    source: 'desktop',
    started_at: 1,
    title: 'stored',
    tool_call_count: 0,
    ...overrides
  }
}

function BranchHarness({
  activeSessionId = null,
  navigate = vi.fn(),
  onCurrentReady,
  onReady,
  requestGateway,
  selectedStoredSessionId = null
}: {
  activeSessionId?: string | null
  navigate?: ReturnType<typeof vi.fn>
  onCurrentReady?: (branchCurrentSession: (messageId?: string) => Promise<boolean>) => void
  onReady: (branchStoredSession: (storedSessionId: string, sessionProfile?: string | null) => Promise<boolean>) => void
  requestGateway: <T>(method: string, params?: Record<string, unknown>) => Promise<T>
  selectedStoredSessionId?: string | null
}) {
  const ref = <T,>(value: T): MutableRefObject<T> => ({ current: value })
  const activeSessionIdRef = ref<string | null>(activeSessionId)
  const selectedStoredSessionIdRef = ref<string | null>(selectedStoredSessionId)

  const actions = useSessionActions({
    activeSessionId,
    activeSessionIdRef,
    busyRef: ref(false),
    creatingSessionRef: ref(false),
    ensureSessionState: () => ({}) as ClientSessionState,
    getRouteToken: () => 'token',
    getRoutedStoredSessionId: () => null,
    navigate: navigate as never,
    requestGateway,
    resetViewSync: vi.fn(),
    routedSessionId: null,
    runtimeIdByStoredSessionIdRef: ref(new Map<string, string>()),
    selectedStoredSessionId,
    selectedStoredSessionIdRef,
    sessionStateByRuntimeIdRef: ref(new Map<string, ClientSessionState>()),
    syncSessionStateToView: vi.fn(),
    updateSessionState: () => ({}) as ClientSessionState
  })

  useEffect(() => {
    onReady(actions.branchStoredSession)
    onCurrentReady?.(actions.branchCurrentSession)
  }, [actions.branchCurrentSession, actions.branchStoredSession, onCurrentReady, onReady])

  return null
}

describe('forkBranch approval-chip owner scoping', () => {
  beforeEach(() => {
    $approvalModes.set({})
  })

  afterEach(() => {
    cleanup()
    setSessions([])
    $sessionTiles.set([])
    setSelectedStoredSessionId(null)
    vi.restoreAllMocks()
  })

  it("keeps the active profile's approval chip when branching a chat another profile owns", async () => {
    // All-profiles view: the open chat lives on `bot` while the gateway (and
    // the status-bar chip keyed by its name) stays on `default`.
    $activeGatewayProfile.set('default')
    reconcileApprovalModeForProfile('default', 'manual')
    setSessions([storedSession({ connection_id: 'local', id: 'stored-parent', message_count: 4, profile: 'bot' })])
    setMessages([{ id: 'tail-user', role: 'user', parts: [{ type: 'text', text: 'question' }] }])

    vi.mocked(requestGatewayForAgent).mockImplementation((async (
      _connectionId: string,
      _profile: string,
      method: string
    ) =>
      method === 'session.branch_whole'
        ? {
            session_id: 'branch-runtime',
            stored_session_id: 'branch-stored',
            title: 'Branch',
            message_count: 4,
            messages_omitted: true,
            // A live parent's branch carries the full runtime info: the bot
            // profile's own approvals.mode.
            info: { approval_mode: 'off' }
          }
        : {}) as never)

    let branchCurrentSession: ((messageId?: string) => Promise<boolean>) | null = null
    render(
      <BranchHarness
        activeSessionId="live-parent"
        onCurrentReady={branch => (branchCurrentSession = branch)}
        onReady={() => undefined}
        requestGateway={vi.fn(async () => ({}) as never)}
        selectedStoredSessionId="stored-parent"
      />
    )
    await waitFor(() => expect(branchCurrentSession).not.toBeNull())

    try {
      await expect(branchCurrentSession!()).resolves.toBe(true)
    } finally {
      vi.mocked(requestGatewayForAgent).mockReset()
    }

    expect(approvalModeForProfile('default')).toBe('manual')
  })
})
