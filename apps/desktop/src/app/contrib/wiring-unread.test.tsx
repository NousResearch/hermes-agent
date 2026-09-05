import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { type ReactElement, StrictMode, useContext } from 'react'
import { MemoryRouter, useLocation, useNavigate } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// Adapted from independent review A/B's mounted repros for #93007. Keep the
// router, unread/owner stores, layout and openSession real; stub backend I/O
// and unrelated feature hooks/surfaces, not the consumer under test.
const h = vi.hoisted(() => {
  const noop = () => undefined

  return { api: new Proxy({}, { get: () => noop }), getSession: vi.fn(), noop, routes: vi.fn() }
})

vi.mock('@/hermes', async original => ({ ...(await original<typeof Hermes>()), getSession: h.getSession }))
vi.mock('@/store/profile', async original => ({ ...(await original<typeof Profile>()), refreshActiveProfile: h.noop }))
vi.mock('@/app/gateway/hooks/use-gateway-boot', () => ({ useGatewayBoot: h.noop }))
vi.mock('@/app/contrib/hooks/use-background-sync', () => ({ useBackgroundSync: h.noop }))
vi.mock('@/app/contrib/hooks/use-desktop-integrations', () => ({ useDesktopIntegrations: h.noop }))
vi.mock('@/app/contrib/hooks/use-pet-bridge', () => ({ usePetBridge: h.noop }))
vi.mock('@/app/contrib/hooks/use-quick-entry-bridge', () => ({ useQuickEntryBridge: h.noop }))
vi.mock('@/app/contrib/hooks/use-session-tile-delegate', () => ({ useSessionTileDelegate: h.noop }))
vi.mock('@/app/session/hooks/use-context-suggestions', () => ({ useContextSuggestions: h.noop }))
vi.mock('@/app/session/hooks/use-background-queue-drain', () => ({ useBackgroundQueueDrain: h.noop }))
vi.mock('@/app/session/hooks/use-route-resume', () => ({ useRouteResume: h.noop }))
vi.mock('@/app/session/hooks/use-session-actions', () => ({ useSessionActions: () => h.api }))
vi.mock('@/app/session/hooks/use-session-list-actions', () => ({ useSessionListActions: () => h.api }))
vi.mock('@/app/session/hooks/use-hermes-config', () => ({ useHermesConfig: () => h.api }))
vi.mock('@/app/session/hooks/use-model-controls', () => ({ useModelControls: () => h.api }))
vi.mock('@/app/session/hooks/use-message-stream', () => ({ useMessageStream: () => h.api }))
vi.mock('@/app/session/hooks/use-preview-routing', () => ({ usePreviewRouting: () => h.api }))
vi.mock('@/app/session/hooks/use-prompt-actions', () => ({ usePromptActions: () => h.api }))
vi.mock('@/app/chat/hooks/use-composer-actions', () => ({ useComposerActions: () => h.api }))
vi.mock('@/app/hooks/use-config-record', () => ({ useHermesConfigRecord: () => ({ isPending: true }) }))
vi.mock('@/app/hooks/use-keybinds', () => ({ useKeybinds: h.noop }))
vi.mock('@/app/hud/handoff', () => ({ useHudHandoff: h.noop }))
vi.mock('@/app/contrib/dev/credits-notice-demo', () => ({ installCreditsNoticeDemo: h.noop }))
vi.mock('@/components/boot-failure-overlay', () => ({ BootFailureOverlay: () => null }))
vi.mock('@/components/confirm-host', () => ({ ConfirmHost: () => null }))
vi.mock('@/components/desktop-install-overlay', () => ({ DesktopInstallOverlay: () => null }))
vi.mock('@/components/find-bar', () => ({ FindBar: () => null }))
vi.mock('@/components/gateway-connecting-overlay', () => ({ GatewayConnectingOverlay: () => null }))
vi.mock('@/components/notifications', () => ({ NotificationStack: () => null }))
vi.mock('@/components/onboarding', () => ({ DesktopOnboardingOverlay: () => null }))
vi.mock('@/components/pet/floating-pet', () => ({ FloatingPet: () => null }))
vi.mock('@/components/remote-display-banner', () => ({ RemoteDisplayBanner: () => null }))
vi.mock('@/components/send-diagnostics-dialog', () => ({ SendDiagnosticsHost: () => null }))
vi.mock('@/components/tips', () => ({ TipHost: () => null }))
vi.mock('@/app/command-palette', () => ({ CommandPalette: () => null }))
vi.mock('@/app/model-picker-overlay', () => ({ ModelPickerOverlay: () => null }))
vi.mock('@/app/model-visibility-overlay', () => ({ ModelVisibilityOverlay: () => null }))
vi.mock('@/app/pet-generate/pet-generate-overlay', () => ({ PetGenerateOverlay: () => null }))
vi.mock('@/app/right-sidebar/file-actions', () => ({ FileActionDialogs: () => null }))
vi.mock('@/app/right-sidebar/files/remote-picker', () => ({ RemoteFolderPicker: () => null }))
vi.mock('@/app/right-sidebar/terminal/persistent', () => ({ PersistentTerminal: () => null }))
vi.mock('@/app/session-picker-overlay', () => ({ SessionPickerOverlay: () => null }))
vi.mock('@/app/session-switcher', () => ({ SessionSwitcher: () => null }))
vi.mock('@/app/settings/plugin-install-modal', () => ({ PluginInstallModal: () => null }))
vi.mock('@/app/shell/titlebar-controls', () => ({ TitlebarControls: () => null }))
vi.mock('@/app/updates-overlay', () => ({ UpdatesOverlay: () => null }))
vi.mock('@/app/contrib/mcp-install-deeplink-dialog', () => ({ McpInstallDeepLinkDialog: () => null }))
vi.mock('@/app/contrib/surfaces', () => ({
  ChatRoutesSurface: () => null,
  SidebarSurface: () => null,
  StatusbarSurface: () => null,
  TerminalSurface: () => null
}))

import { group, split } from '@/components/pane-shell/tree/model'
import {
  $activeTreeGroup,
  $hiddenTreePanes,
  $layoutTree,
  noteActiveTreeGroup
} from '@/components/pane-shell/tree/store'
import { setWorkspaceScope } from '@/components/pane-shell/workspace-scope'
import { registry } from '@/contrib/registry'
import type * as Hermes from '@/hermes'
import type * as Profile from '@/store/profile'
import { $activeGatewayProfile, $showAllProfiles } from '@/store/profile'
import {
  $connection,
  $messagingSessions,
  $selectedStoredSessionId,
  $sessionResumeRequest,
  $sessions,
  $unreadFinishedSessionIds,
  _resetSessionOwnerHintsForTests,
  getSessionOwnerHint,
  requestSessionResume,
  setSessionOwnerHint
} from '@/store/session'
import { $unreadSessionTargets } from '@/store/session-dot-state'
import { $focusedStoredSessionId, $sessionTiles, clearAllSessionStates } from '@/store/session-states'
import { $openNextUnreadRequest, requestOpenNextUnread } from '@/store/session-unread-navigation'
import { $unreadWriteGuard } from '@/store/session-unread-remote'
import type { SessionInfo } from '@/types/hermes'

import { sessionRoute } from '../routes'

import { ContribWiringContext } from './context'
import type { SidebarActions } from './types'
import { ContribWiring } from './wiring'

const row = (id = 'unread', extra: Partial<SessionInfo> = {}): SessionInfo => ({
  id,
  profile: 'ops',
  connection_id: 'source-a',
  source: 'desktop',
  title: id,
  ended_at: null,
  input_tokens: 0,
  output_tokens: 0,
  is_active: false,
  model: null,
  preview: null,
  tool_call_count: 0,
  started_at: 1,
  last_active: 100,
  message_count: 2,
  unread: true,
  ...extra
})

function Controls() {
  const location = useLocation()
  const navigate = useNavigate()
  const api = useContext(ContribWiringContext)!
  const actions = (api.sidebar as ReactElement<{ actions: SidebarActions }>).props.actions

  return (
    <>
      <output data-testid="route">{location.pathname}</output>
      <button onClick={requestOpenNextUnread}>Next unread</button>
      <button onClick={() => navigate(sessionRoute('manual'))}>Manual selection</button>
      <button onClick={() => navigate(sessionRoute('initial'))}>Return</button>
      <button onClick={() => actions.onResumeSession('unread', row('unread', { connection_id: undefined }))}>
        Ordinary row
      </button>
    </>
  )
}

const mount = () =>
  render(
    <StrictMode>
      <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
        <MemoryRouter initialEntries={[sessionRoute('initial')]}>
          <ContribWiring>
            <Controls />
          </ContribWiring>
        </MemoryRouter>
      </QueryClientProvider>
    </StrictMode>
  )

const clickCount = () => fireEvent.click(screen.getByText('Next unread'))

const flush = async () =>
  act(async () => {
    await Promise.resolve()
  })

const path = () => screen.getByTestId('route').textContent

const setConnectionId = (connectionId: string) =>
  $connection.set({
    connectionId,
    baseUrl: 'http://unused.invalid',
    wsUrl: 'ws://unused.invalid',
    token: '',
    isFullscreen: false,
    nativeOverlayWidth: 0,
    logs: [],
    windowButtonPosition: null
  })

function deferred() {
  let resolve!: (value?: unknown) => void

  const promise = new Promise(done => {
    resolve = done
  })

  return { promise, resolve }
}

const disposers: (() => void)[] = []

beforeEach(() => {
  vi.clearAllMocks()
  h.getSession.mockReset()
  h.routes.mockReset()
  window.hermesDesktop = { getProfileRoutes: h.routes } as unknown as typeof window.hermesDesktop
  clearAllSessionStates()
  $sessionTiles.set([])
  _resetSessionOwnerHintsForTests()
  $openNextUnreadRequest.set(0)
  setConnectionId('primary')
  $activeGatewayProfile.set('default')
  $showAllProfiles.set(true)
  setWorkspaceScope('sessions')
  $selectedStoredSessionId.set('initial')
  $sessionResumeRequest.set(null)
  $sessions.set([row()])
  $messagingSessions.set([])
  $unreadFinishedSessionIds.set(['unread'])
  $unreadWriteGuard.set(new Map())
  $hiddenTreePanes.set(new Set())

  for (const id of ['sessions', 'workspace', 'session-tile:unread', 'session-tile:root']) {
    disposers.push(registry.register({ area: 'panes', id, data: { placement: 'main' }, render: () => null }))
  }

  $layoutTree.set(split('row', [group(['sessions']), group(['workspace'])]))
  $activeTreeGroup.set(null)
  h.routes.mockResolvedValue([])
  h.getSession.mockResolvedValue({ session: row(), messages: [] })
})
afterEach(() => {
  cleanup()
  disposers.splice(0).forEach(dispose => dispose())
})

describe('mounted unread intent and routing', () => {
  it.each([
    'route',
    'roundtrip',
    'selection',
    'resume',
    'focus',
    'workspace',
    'workspace-owner',
    'connection',
    'profile',
    'filter'
  ])('does not publish a delayed preflight after newer %s intent', async change => {
    const pending = deferred()
    h.getSession.mockReturnValue(pending.promise)

    if (change === 'workspace-owner') {
      setWorkspaceScope('bots', 'source-a::ops')
    }

    mount()
    clickCount()
    await flush()
    expect(h.getSession).toHaveBeenCalledOnce()
    act(() => {
      if (change === 'route' || change === 'roundtrip') {
        fireEvent.click(screen.getByText('Manual selection'))
      }

      if (change === 'selection') {
        $selectedStoredSessionId.set('manual')
      }

      if (change === 'resume') {
        requestSessionResume('manual')
      }

      if (change === 'focus') {
        const tileGroup = group(['session-tile:manual'])
        $layoutTree.set(split('row', [group(['workspace']), tileGroup]))
        noteActiveTreeGroup(tileGroup.id)
        expect($focusedStoredSessionId.get()).toBe('manual')
      }

      if (change === 'workspace' || change === 'workspace-owner') {
        setWorkspaceScope('bots', 'source-b::ops')
      }

      if (change === 'connection') {
        setConnectionId('other')
      }

      if (change === 'profile') {
        $activeGatewayProfile.set('other')
      }

      if (change === 'filter') {
        $showAllProfiles.set(false)
      }
    })

    if (change === 'roundtrip') {
      fireEvent.click(screen.getByText('Return'))
    }

    const resume = $sessionResumeRequest.get()
    await act(async () => pending.resolve({ session: row() }))
    expect(path()).toBe(sessionRoute(change === 'route' ? 'manual' : 'initial'))
    expect($sessionResumeRequest.get()).toBe(resume)
    expect($unreadFinishedSessionIds.get()).toContain('unread')
    expect(getSessionOwnerHint('unread')).toBeUndefined()
  })

  it('stops the probe ladder if a connection changes during registry lookup', async () => {
    const pending = deferred()
    h.routes.mockReturnValue(pending.promise)
    mount()
    clickCount()
    act(() => setConnectionId('other'))
    await act(async () => pending.resolve([]))
    expect(h.getSession).not.toHaveBeenCalled()
    expect(path()).toBe('/initial')
  })

  it('can accept a fresh intent while an invalidated request is still pending', async () => {
    const old = deferred()
    const fresh = deferred()
    h.getSession.mockReturnValueOnce(old.promise).mockReturnValueOnce(fresh.promise)
    mount()
    clickCount()
    await flush()
    fireEvent.click(screen.getByText('Manual selection'))
    clickCount()
    await flush()
    expect(h.getSession).toHaveBeenCalledTimes(2)
    await act(async () => old.resolve())
    clickCount()
    await flush()
    expect(h.getSession).toHaveBeenCalledTimes(2)
    await act(async () => fresh.resolve())
    expect(path()).toBe('/unread')
  })

  it.each(['read', 'archive', 'remove'])('revalidates live target membership after %s', async change => {
    const pending = deferred()
    h.getSession.mockReturnValue(pending.promise)
    mount()
    clickCount()
    await flush()
    act(() => {
      $sessions.set(
        change === 'remove' ? [] : [row('unread', change === 'read' ? { unread: false } : { archived: true })]
      )
      $unreadFinishedSessionIds.set([])
    })
    expect($unreadSessionTargets.get()).toEqual([])
    await act(async () => pending.resolve())
    expect(path()).toBe('/initial')
    expect($sessionResumeRequest.get()).toBeNull()
  })

  it.each(['Next unread', 'Ordinary row'])(
    '%s clears a migrated owner hint and replaces the old resume intent',
    async button => {
      $sessions.set([row('unread', { connection_id: undefined })])
      requestSessionResume('unread', { connectionId: 'stale-local', profile: 'ops' })
      const previous = $sessionResumeRequest.get()
      mount()
      await act(async () => fireEvent.click(screen.getByText(button)))
      expect(path()).toBe('/unread')
      expect(getSessionOwnerHint('unread')).toBeUndefined()
      expect($sessionResumeRequest.get()).toMatchObject({ sessionId: 'unread' })
      expect($sessionResumeRequest.get()?.ownerRoute).toBeUndefined()
      expect($sessionResumeRequest.get()).not.toBe(previous)
    }
  )

  it('coalesces pending count clicks and preserves exact alias routing', async () => {
    const pending = deferred()
    h.getSession.mockReturnValue(pending.promise)
    h.routes.mockResolvedValue([{ connectionId: 'source-a', profile: 'ops', targetProfile: 'backend-ops' }])
    mount()
    clickCount()
    await flush()
    clickCount()
    expect(h.getSession).toHaveBeenCalledOnce()
    expect(h.getSession).toHaveBeenCalledWith('unread', { connectionId: 'source-a', profile: 'backend-ops' })
    await act(async () => pending.resolve())
    expect($sessionResumeRequest.get()?.ownerRoute).toEqual({
      connectionId: 'source-a',
      profile: 'ops',
      targetProfile: 'backend-ops'
    })
    expect(path()).toBe('/unread')
  })

  it('does not replay or accept late completion across a real remount', async () => {
    const pending = deferred()
    h.getSession.mockReturnValue(pending.promise)
    const first = mount()
    clickCount()
    await flush()
    first.unmount()
    mount()
    await act(async () => pending.resolve())
    expect(h.getSession).toHaveBeenCalledOnce()
    expect(path()).toBe('/initial')
    expect($unreadFinishedSessionIds.get()).toContain('unread')
  })

  it('retains the primary profile-first ladder and exact registry fallback', async () => {
    $sessions.set([row('unread', { connection_id: undefined })])
    h.routes.mockResolvedValue([{ connectionId: 'primary', profile: 'ops', targetProfile: 'backend-ops' }])
    h.getSession.mockRejectedValueOnce(new Error('profile route unavailable'))
    mount()
    await act(async () => clickCount())
    expect(h.getSession.mock.calls).toEqual([
      ['unread', 'ops'],
      ['unread', { connectionId: 'primary', profile: 'backend-ops' }]
    ])
    expect($sessionResumeRequest.get()?.ownerRoute).toEqual({
      connectionId: 'primary',
      profile: 'ops',
      targetProfile: 'backend-ops'
    })
    expect(path()).toBe('/unread')
  })

  it('does not acknowledge an already-selected main ID whose hydrated owner is unproven', async () => {
    $selectedStoredSessionId.set('unread')
    $unreadFinishedSessionIds.set(['unread'])
    mount()
    await act(async () => clickCount())
    expect(path()).toBe('/initial')
    expect($sessionResumeRequest.get()).toBeNull()
    expect($unreadFinishedSessionIds.get()).toContain('unread')
  })

  it('selects the unread source, not its newer read twin', async () => {
    $sessions.set([row(), row('unread', { connection_id: 'source-b', last_active: 200, unread: false })])
    $unreadFinishedSessionIds.set([])
    mount()
    await act(async () => clickCount())
    expect(h.getSession.mock.calls).toEqual([['unread', { connectionId: 'source-a', profile: 'ops' }]])
    expect($sessionResumeRequest.get()?.ownerRoute?.connectionId).toBe('source-a')
  })

  it.each(['wrong-owner', 'unknown-owner', 'ownerless-probe', 'lineage'])(
    'skips an ambiguous open tile (%s) without focus, ack, or owner mutation',
    async kind => {
      const tileId = kind === 'lineage' ? 'root' : 'unread'
      const ownerRoute = kind === 'unknown-owner' ? undefined : { connectionId: 'source-b', profile: 'ops' }
      $sessionTiles.set([{ storedSessionId: tileId, ownerRoute, workspaceMode: 'sessions' }])

      if (kind === 'ownerless-probe') {
        $sessions.set([row('unread', { connection_id: undefined })])
      }

      if (kind === 'lineage') {
        $sessions.set([row('unread', { _lineage_root_id: 'root' })])
      }

      $layoutTree.set(split('row', [group(['sessions']), group(['workspace', `session-tile:${tileId}`])]))
      $unreadFinishedSessionIds.set(['unread'])
      const tiles = $sessionTiles.get()
      const currentHint = { connectionId: 'source-b', profile: 'ops' }
      setSessionOwnerHint('unread', currentHint)
      mount()
      expect($unreadFinishedSessionIds.get()).toContain('unread')
      await act(async () => clickCount())
      expect(path()).toBe('/initial')
      expect($activeTreeGroup.get()).toBeNull()
      expect($focusedStoredSessionId.get()).toBe('initial')
      expect($unreadFinishedSessionIds.get()).toContain('unread')
      expect($sessionResumeRequest.get()).toBeNull()
      expect($sessionTiles.get()).toBe(tiles)
      expect(getSessionOwnerHint('unread')).toEqual(currentHint)
    }
  )

  it('still focuses a tile whose exact owner matches the preflight', async () => {
    $sessionTiles.set([
      { storedSessionId: 'unread', ownerRoute: { connectionId: 'source-a', profile: 'ops' }, workspaceMode: 'sessions' }
    ])
    $layoutTree.set(split('row', [group(['sessions']), group(['workspace', 'session-tile:unread'])]))
    mount()
    await act(async () => clickCount())
    expect(path()).toBe('/initial')
    expect($focusedStoredSessionId.get()).toBe('unread')
    expect($unreadFinishedSessionIds.get()).not.toContain('unread')
  })

  it('falls through an ambiguous tile and a failed owner without acknowledging either', async () => {
    $sessionTiles.set([{ storedSessionId: 'unread', ownerRoute: { connectionId: 'source-b', profile: 'ops' } }])
    $sessions.set([row(), row('offline', { last_active: 50 }), row('available', { last_active: 10 })])
    $unreadFinishedSessionIds.set(['unread', 'offline', 'available'])
    h.getSession.mockImplementation(async id => {
      if (id === 'offline') {
        throw new Error('offline')
      }

      return { session: row(id) }
    })
    mount()
    await act(async () => clickCount())
    expect(path()).toBe('/available')
    expect($unreadFinishedSessionIds.get()).toEqual(['unread', 'offline'])
  })
})
