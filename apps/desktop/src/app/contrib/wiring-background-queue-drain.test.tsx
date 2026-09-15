import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, render } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import {
  $parkedQueueSessions,
  $queuedPromptsBySession,
  enqueueQueuedPrompt,
  getQueuedPrompts
} from '@/store/composer-queue'
import { $activeConnectionId } from '@/store/connections'
import { $activeSessionId, $connection, $gatewayState, $selectedStoredSessionId, setSessions } from '@/store/session'
import { clearAllSessionStates } from '@/store/session-states'
import { makeSessionInfo } from '@/test/session-info'

import type { SubmitTextOptions } from '../session/hooks/use-prompt-actions/utils'

import type { AmbientGatewayRequest } from './session-rpc-dispatcher'

// Mount the actual controller: its gateway subscription and enabled policy are
// NOT copied into a test harness. Keep the real queue hook/stores and session
// dispatcher. Stub unrelated chrome/boot and the prompt/network boundaries;
// this proves scheduling/routing, not a WebSocket or an LLM's execution.
const fixture = vi.hoisted(() => ({
  ambientRequest: vi.fn(async () => {
    throw new Error('foreground must not receive background RPCs')
  }),
  remoteRequest: vi.fn(async () => ({ accepted: true })),
  submitText: vi.fn<(text: string, options?: SubmitTextOptions) => Promise<boolean>>(),
  requestGateway: null as AmbientGatewayRequest | null,
  runtimeMap: { current: new Map([['stored-remote', 'rt-remote']]) },
  stateMap: { current: new Map() },
  selectedRef: { current: 'stored-foreground' },
  activeRef: { current: 'rt-foreground' }
}))

vi.mock('@/store/gateway', async importActual => ({
  ...(await importActual<Record<string, unknown>>()),
  activeGatewayConnectionId: () => 'local',
  requestGatewayForAgent: fixture.remoteRequest
}))
vi.mock('../gateway/hooks/use-gateway-request', () => ({
  useGatewayRequest: () => ({
    connectionRef: { current: null },
    gateway: null,
    gatewayRef: { current: null },
    requestGateway: fixture.ambientRequest
  })
}))
vi.mock('../session/hooks/use-session-state-cache', () => ({
  useSessionStateCache: () => ({
    activeSessionIdRef: fixture.activeRef,
    selectedStoredSessionIdRef: fixture.selectedRef,
    runtimeIdByStoredSessionIdRef: fixture.runtimeMap,
    sessionStateByRuntimeIdRef: fixture.stateMap
  })
}))
vi.mock('../session/hooks/use-prompt-actions', () => ({
  usePromptActions: ({ requestGateway }: { requestGateway: AmbientGatewayRequest }) => {
    fixture.requestGateway = requestGateway
    fixture.submitText.mockImplementation(async (text, options) => {
      await requestGateway('prompt.submit', { session_id: options?.sessionId, text })

      return true
    })

    return { submitText: fixture.submitText }
  }
}))
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
vi.mock('../command-palette', () => ({ CommandPalette: () => null }))
vi.mock('../model-picker-overlay', () => ({ ModelPickerOverlay: () => null }))
vi.mock('../model-visibility-overlay', () => ({ ModelVisibilityOverlay: () => null }))
vi.mock('../pet-generate/pet-generate-overlay', () => ({ PetGenerateOverlay: () => null }))
vi.mock('../right-sidebar/file-actions', () => ({ FileActionDialogs: () => null }))
vi.mock('../right-sidebar/files/remote-picker', () => ({ RemoteFolderPicker: () => null }))
vi.mock('../right-sidebar/terminal/persistent', () => ({ PersistentTerminal: () => null }))
vi.mock('../session-import', () => ({ SessionImportView: () => null }))
vi.mock('../session-picker-overlay', () => ({ SessionPickerOverlay: () => null }))
vi.mock('../session-switcher', () => ({ SessionSwitcher: () => null }))
vi.mock('../settings/plugin-install-modal', () => ({ PluginInstallModal: () => null }))
vi.mock('../shell/titlebar-controls', () => ({ TitlebarControls: () => null }))
vi.mock('../updates-overlay', () => ({ UpdatesOverlay: () => null }))
vi.mock('./mcp-install-deeplink-dialog', () => ({ McpInstallDeepLinkDialog: () => null }))
vi.mock('./surfaces', () => ({
  ChatRoutesSurface: () => null,
  SidebarSurface: () => null,
  StatusbarSurface: () => null,
  TerminalSurface: () => null
}))
vi.mock('../gateway/hooks/use-gateway-boot', () => ({ useGatewayBoot: () => ({}) }))
vi.mock('../session/hooks/use-context-suggestions', () => ({ useContextSuggestions: () => ({}) }))
vi.mock('../session/hooks/use-cwd-actions', () => ({ useCwdActions: () => ({}) }))
vi.mock('../session/hooks/use-hermes-config', () => ({ useHermesConfig: () => ({}) }))
vi.mock('../session/hooks/use-message-stream', () => ({ useMessageStream: () => ({}) }))
vi.mock('../session/hooks/use-model-controls', () => ({ useModelControls: () => ({}) }))
vi.mock('../session/hooks/use-preview-routing', () => ({ usePreviewRouting: () => ({}) }))
vi.mock('../session/hooks/use-route-resume', () => ({ useRouteResume: () => ({}) }))
vi.mock('../session/hooks/use-session-actions', () => ({ useSessionActions: () => ({}) }))
vi.mock('../session/hooks/use-session-list-actions', () => ({ useSessionListActions: () => ({}) }))
vi.mock('../chat/hooks/use-composer-actions', () => ({ useComposerActions: () => ({}) }))
vi.mock('../hooks/use-keybinds', () => ({ useKeybinds: () => ({}) }))
vi.mock('../hud/handoff', () => ({ useHudHandoff: () => ({}) }))
vi.mock('../shell/hooks/use-overlay-routing', () => ({ useOverlayRouting: () => ({}) }))
vi.mock('./hooks/use-background-sync', () => ({ useBackgroundSync: () => ({}) }))
vi.mock('./hooks/use-desktop-integrations', () => ({ useDesktopIntegrations: () => ({}) }))
vi.mock('./hooks/use-pet-bridge', () => ({ usePetBridge: () => ({}) }))
vi.mock('./hooks/use-quick-entry-bridge', () => ({ useQuickEntryBridge: () => ({}) }))
vi.mock('./hooks/use-session-tile-delegate', () => ({ useSessionTileDelegate: () => ({}) }))
vi.mock('@/themes/use-skin-command', () => ({ useSkinCommand: () => ({}) }))
vi.mock('../hooks/use-config-record', () => ({ useHermesConfigRecord: () => ({}) }))
vi.mock('./dev/credits-notice-demo', () => ({ installCreditsNoticeDemo: () => undefined }))
vi.mock('@/store/wake-word', async importActual => ({
  ...(await importActual<Record<string, unknown>>()),
  armWakeWord: vi.fn()
}))
vi.mock('./panes', async () => {
  const { atom } = await import('nanostores')

  return { $restartPreviewServer: atom(null), useTitlebarToolContributions: () => [] }
})
vi.mock('../shell/hooks/use-window-controls-overlay-width', () => ({ useWindowControlsOverlayWidth: () => 0 }))

import { ContribWiring } from './wiring'

const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })

beforeEach(() => {
  vi.useFakeTimers()
  vi.clearAllMocks()
  clearAllSessionStates()
  $queuedPromptsBySession.set({})
  $parkedQueueSessions.set({})
  $connection.set({
    connectionId: 'local',
    profile: 'default',
    mode: 'local',
    baseUrl: 'http://foreground.invalid',
    wsUrl: 'ws://foreground.invalid',
    token: '',
    logs: [],
    isFullscreen: false,
    nativeOverlayWidth: 0,
    windowButtonPosition: null
  })
  $activeSessionId.set('rt-foreground')
  $selectedStoredSessionId.set('stored-foreground')
  setSessions([
    makeSessionInfo({ id: 'stored-foreground', connection_id: 'local', profile: 'default' }),
    makeSessionInfo({ id: 'stored-remote', connection_id: 'remote-healthy', profile: 'worker' })
  ])
})

afterEach(() => {
  cleanup()
  queryClient.clear()
  $queuedPromptsBySession.set({})
  $parkedQueueSessions.set({})
  setSessions([])
  clearAllSessionStates()
  $activeSessionId.set(null)
  $selectedStoredSessionId.set(null)
  $connection.set(null)
  $gatewayState.set('closed')
  fixture.requestGateway = null
  vi.useRealTimers()
})

describe('ContribWiring background queue: independent remote owner', () => {
  it.each(['open', 'closed'] as const)('submits healthy remote queue with foreground %s', async foregroundState => {
    $gatewayState.set(foregroundState)
    render(
      <MemoryRouter>
        <QueryClientProvider client={queryClient}>
          <ContribWiring>{null}</ContribWiring>
        </QueryClientProvider>
      </MemoryRouter>
    )

    // Positive reachability probe through the controller's exact dispatcher,
    // with the foreground state already set. Only scheduling differs below.
    await expect(fixture.requestGateway!('session.info', { session_id: 'rt-remote' })).resolves.toEqual({
      accepted: true
    })
    expect(fixture.remoteRequest).toHaveBeenCalledWith('remote-healthy', 'worker', 'session.info', {
      session_id: 'rt-remote'
    })
    fixture.remoteRequest.mockClear()

    await act(async () => {
      enqueueQueuedPrompt('stored-remote', { text: 'continue remotely', attachments: [] })
    })
    await act(async () => {
      await vi.advanceTimersByTimeAsync(3000)
    })

    expect(fixture.ambientRequest).not.toHaveBeenCalled()
    expect($gatewayState.get()).toBe(foregroundState)
    expect($activeConnectionId.get()).toBe('local')
    expect.soft(fixture.submitText).toHaveBeenCalledExactlyOnceWith('continue remotely', {
      attachments: [],
      fromQueue: true,
      sessionId: 'rt-remote',
      storedSessionId: 'stored-remote'
    })
    expect.soft(fixture.remoteRequest).toHaveBeenCalledExactlyOnceWith('remote-healthy', 'worker', 'prompt.submit', {
      session_id: 'rt-remote',
      text: 'continue remotely'
    })
    expect(getQueuedPrompts('stored-remote')).toHaveLength(0)
  })
})
