// @vitest-environment jsdom
import { useStore } from '@nanostores/react'
import { act, cleanup, render, waitFor } from '@testing-library/react'
import { useEffect, useRef } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { getHermesConfig } from '@/hermes'
import { persistString } from '@/lib/storage'
import { $projectScope, $projectTree, ALL_PROJECTS } from '@/store/projects'
import {
  $activeSessionId,
  $currentCwd,
  $freshDraftReady,
  setActiveSessionId,
  setCurrentCwd,
  setFreshDraftReady,
  setNewChatWorkspaceTarget
} from '@/store/session'

import type { ClientSessionState } from '../../types'

import { useBackgroundSync } from '../../contrib/hooks/use-background-sync'
import { useHermesConfig } from './use-hermes-config'
import { useSessionActions } from './use-session-actions'

vi.mock('@/hermes', () => ({
  getHermesConfig: vi.fn(),
  getHermesConfigDefaults: vi.fn().mockResolvedValue({}),
  deleteSession: vi.fn(),
  getSessionMessages: vi.fn(),
  listAllProfileSessions: vi.fn(),
  setApiRequestProfile: vi.fn(),
  setSessionArchived: vi.fn()
}))

const WORKSPACE_CWD_KEY = 'hermes.desktop.workspace-cwd'
const PROJECT_CWD = 'C:\\Users\\example\\work\\project'
const CONFIGURED_HOME_CWD = 'C:\\Users\\example'

type DraftHandle = Pick<ReturnType<typeof useSessionActions>, 'startFreshSessionDraft'>

/**
 * Composes the real fresh-draft action with background-sync config refresh —
 * the seam #65274 regressed: startFreshSessionDraft clears the session ref and
 * seeds project cwd, then useBackgroundSync reloads config.yaml terminal.cwd.
 */
function FreshDraftConfigSyncHarness({
  gatewayState,
  onReady
}: {
  gatewayState: string
  onReady: (handle: DraftHandle) => void
}) {
  const activeSessionIdRef = useRef<string | null>('prior-session')
  const busyRef = useRef(false)
  const creatingSessionRef = useRef(false)
  const selectedStoredSessionIdRef = useRef<string | null>('stored-prior')
  const runtimeIdByStoredSessionIdRef = useRef(new Map<string, string>())
  const sessionStateByRuntimeIdRef = useRef(new Map<string, ClientSessionState>())

  const activeSessionId = useStore($activeSessionId)
  const freshDraftReady = useStore($freshDraftReady)

  const { refreshHermesConfig } = useHermesConfig({
    activeSessionIdRef,
    refreshProjectBranch: vi.fn().mockResolvedValue(undefined)
  })

  const actions = useSessionActions({
    activeSessionId,
    activeSessionIdRef,
    busyRef,
    creatingSessionRef,
    ensureSessionState: () => ({}) as ClientSessionState,
    getRouteToken: () => 'token',
    getRoutedStoredSessionId: () => selectedStoredSessionIdRef.current,
    navigate: vi.fn() as never,
    requestGateway: vi.fn().mockResolvedValue({}),
    resetViewSync: vi.fn(),
    runtimeIdByStoredSessionIdRef,
    selectedStoredSessionId: selectedStoredSessionIdRef.current,
    selectedStoredSessionIdRef,
    sessionStateByRuntimeIdRef,
    syncSessionStateToView: vi.fn(),
    updateSessionState: () => ({}) as ClientSessionState
  })

  useBackgroundSync({
    activeGatewayProfile: 'default',
    activeIsMessaging: false,
    activeSessionId,
    freshDraftReady,
    gatewayState,
    refreshActiveMessagingTranscript: vi.fn(),
    refreshCronJobs: vi.fn(),
    refreshCurrentModel: vi.fn(),
    refreshHermesConfig,
    refreshMessagingSessions: vi.fn(),
    refreshSessions: vi.fn(),
    requestGateway: vi.fn().mockResolvedValue({})
  })

  useEffect(() => {
    onReady({ startFreshSessionDraft: actions.startFreshSessionDraft })
  }, [actions, onReady])

  return null
}

describe('fresh-draft + background-sync config composition (#65274)', () => {
  beforeEach(() => {
    setActiveSessionId('prior-session')
    setCurrentCwd('C:\\Users\\example\\other-worktree')
    setFreshDraftReady(false)
    setNewChatWorkspaceTarget(undefined)
    persistString(WORKSPACE_CWD_KEY, null)
    $projectTree.set([
      {
        id: 'p_app',
        label: 'App',
        path: PROJECT_CWD,
        repos: [{ groups: [], id: PROJECT_CWD, label: 'project', path: PROJECT_CWD, sessionCount: 0 }],
        sessionCount: 0
      }
    ])
    $projectScope.set('p_app')
    vi.mocked(getHermesConfig).mockResolvedValue({
      terminal: { cwd: CONFIGURED_HOME_CWD }
    } as Awaited<ReturnType<typeof getHermesConfig>>)
  })

  afterEach(() => {
    cleanup()
    setActiveSessionId(null)
    setCurrentCwd('')
    setFreshDraftReady(false)
    setNewChatWorkspaceTarget(undefined)
    $projectScope.set(ALL_PROJECTS)
    $projectTree.set([])
    persistString(WORKSPACE_CWD_KEY, null)
    vi.clearAllMocks()
  })

  it('keeps project cwd after startFreshSessionDraft triggers background config refresh', async () => {
    let handle: DraftHandle | null = null

    render(
      <FreshDraftConfigSyncHarness gatewayState="open" onReady={h => (handle = h)} />
    )
    await waitFor(() => expect(handle).not.toBeNull())

    await act(async () => {
      handle!.startFreshSessionDraft()
    })

    // Draft must clear the active session and seed the project root before the
    // background-sync effect reloads config (terminal.cwd is often Path.home()).
    expect($activeSessionId.get()).toBeNull()
    expect($freshDraftReady.get()).toBe(true)
    expect($currentCwd.get()).toBe(PROJECT_CWD)

    await waitFor(() => {
      expect(getHermesConfig).toHaveBeenCalled()
    })

    // Without $freshDraftReady the null session ref would let terminal.cwd win.
    expect($currentCwd.get()).toBe(PROJECT_CWD)
  })
})
