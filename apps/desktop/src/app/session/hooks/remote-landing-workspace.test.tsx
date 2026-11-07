import { useStore } from '@nanostores/react'
import { act, cleanup, render, screen } from '@testing-library/react'
import { useEffect, useRef } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { browserDefaultProjectDirKey, createBrowserSettings } from '@/lib/browser-settings'
import { createClientSessionState } from '@/lib/chat-runtime'
import { requestGatewayForAgent } from '@/store/gateway'
import { $activeGatewayProfile, $newChatProfile, $newChatRoute } from '@/store/profile'
import { $projectScope, ALL_PROJECTS } from '@/store/project-scope'
import { $projectTree } from '@/store/projects'
import {
  $activeSessionId,
  $connection,
  $currentCwd,
  applyConfiguredDefaultProjectDir,
  ensureDefaultWorkspaceCwd,
  getRememberedWorkspaceCwd,
  setCurrentCwd,
  setNewChatWorkspaceTarget,
  setSelectedStoredSessionId
} from '@/store/session'

import { useSessionActions } from './use-session-actions'

vi.mock('@/store/profile', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  ensureGatewayAgent: vi.fn().mockResolvedValue(undefined),
  ensureGatewayProfile: vi.fn().mockResolvedValue(undefined)
}))

vi.mock('@/store/gateway', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  requestGatewayForAgent: vi.fn(),
  retainGatewayForAgent: vi.fn(async () => () => undefined)
}))

type SessionActions = ReturnType<typeof useSessionActions>
type GatewayRequest = Parameters<typeof useSessionActions>[0]['requestGateway']

interface LandingDraftProps {
  onReady: (actions: SessionActions) => void
  requestGateway: GatewayRequest
}

function LandingDraft({ onReady, requestGateway }: LandingDraftProps) {
  const actions = useSessionActions({
    activeSessionId: null,
    activeSessionIdRef: useRef<null | string>(null),
    busyRef: useRef(false),
    creatingSessionRef: useRef(false),
    ensureSessionState: sessionId => createClientSessionState(sessionId),
    getRouteToken: () => 'landing-draft',
    getRoutedStoredSessionId: () => null,
    navigate: vi.fn(),
    requestGateway,
    resetViewSync: vi.fn(),
    routedSessionId: null,
    runtimeIdByStoredSessionIdRef: useRef(new Map()),
    selectedStoredSessionId: null,
    selectedStoredSessionIdRef: useRef<null | string>(null),
    sessionStateByRuntimeIdRef: useRef(new Map()),
    syncSessionStateToView: vi.fn(),
    updateSessionState: (sessionId, update) => update(createClientSessionState(sessionId))
  })

  useEffect(() => onReady(actions), [actions, onReady])

  return <output data-testid='workspace'>{useStore($currentCwd)}</output>
}

describe('remote landing draft workspace (#148245)', () => {
  const configured = '/srv/configured-project'
  const remembered = '/srv/previous-project'
  const memoryKey = 'hermes.desktop.workspace-cwd.remote.http%3A%2F%2Fbackend-a.default'
  const sanitizeWorkspaceCwd = vi.fn(async () => ({ cwd: 'C:\\native-only-default' }))

  beforeEach(() => {
    vi.clearAllMocks()
    $activeGatewayProfile.set('default')
    $newChatProfile.set(null)
    $newChatRoute.set(null)
    $projectScope.set(ALL_PROJECTS)
    $projectTree.set([])
    $activeSessionId.set(null)
    setSelectedStoredSessionId(null)
    setNewChatWorkspaceTarget(undefined)
    applyConfiguredDefaultProjectDir(null)
    $connection.set({ baseUrl: 'http://backend-a', mode: 'remote', profile: 'default' } as never)
    setCurrentCwd(remembered)
    ;(window as { hermesDesktop?: unknown }).hermesDesktop = {
      sanitizeWorkspaceCwd,
      settings: {
        getDefaultProjectDir: vi.fn(async () => ({ defaultLabel: configured, dir: configured, resolvedCwd: configured }))
      }
    }
  })

  afterEach(() => {
    cleanup()
    applyConfiguredDefaultProjectDir(null)
    $activeSessionId.set(null)
    setSelectedStoredSessionId(null)
    setNewChatWorkspaceTarget(undefined)
    $connection.set(null)
    $currentCwd.set('')
    window.localStorage.removeItem(memoryKey)
    window.localStorage.removeItem(browserDefaultProjectDirKey('/fixture', 'default'))
    window.localStorage.removeItem('hermes:browser:default-project-dir-adopted:["/fixture"]')
    delete (window as { hermesDesktop?: unknown }).hermesDesktop
    vi.restoreAllMocks()
  })

  it.each(['native', 'browser'])('uses the configured %s folder on first send without a New chat reset', async host => {
    if (host === 'browser') {
      const settings = createBrowserSettings({ basePath: '/fixture', currentProfile: () => 'default' })
      await settings.setDefaultProjectDir(configured)
      ;(window as { hermesDesktop?: unknown }).hermesDesktop = { sanitizeWorkspaceCwd, settings }
    }

    const creates: Record<string, unknown>[] = []

    const requestGateway: GatewayRequest = vi.fn(async (method, params) => {
      if (method === 'session.create') {
        creates.push(params ?? {})
      }

      return { session_id: 'first-runtime', stored_session_id: null } as never
    })

    vi.mocked(requestGatewayForAgent).mockImplementation(async (_connectionId, _profile, method, params) =>
      requestGateway(method, params) as never
    )
    let actions: SessionActions | undefined
    render(<LandingDraft onReady={value => { actions = value }} requestGateway={requestGateway} />)

    await act(async () => { await ensureDefaultWorkspaceCwd() })
    await act(async () => { await actions!.createBackendSessionForSend() })

    expect(creates).toHaveLength(1)
    expect(creates[0]).toMatchObject({ cwd: configured, profile: 'default', source: 'desktop' })
    expect(screen.getByTestId('workspace').textContent).toBe(configured)
    expect(getRememberedWorkspaceCwd()).toBe(remembered)
    expect(sanitizeWorkspaceCwd).not.toHaveBeenCalled()
  })
})
