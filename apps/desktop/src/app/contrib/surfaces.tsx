/**
 * Wiring surfaces — each pane is its own memoized component. Every surface
 * reads the reactive state it renders from at the leaf (its own atom
 * subscriptions) and reaches the controller's callbacks through the stable
 * `actions` bag, so a state change scoped to one surface (or a bare
 * wiring-controller tick) never re-renders another. This is what keeps the
 * layout tree's zones independently rendered — the whole point of the shell.
 */

import { useStore } from '@nanostores/react'
import { type ComponentProps, memo, type ReactNode, Suspense, useCallback, useMemo, useState } from 'react'
import { Navigate, Route, Routes, useParams } from 'react-router'

import { ContribBoundary, ContribRender } from '@/contrib/react/boundary'
import { useContributions } from '@/contrib/react/use-contributions'
import { useI18n } from '@/i18n'
import { $activeConnectionId } from '@/store/connections'
import { $gateway, requestGatewayForAgent } from '@/store/gateway'
import { $showsAdvancedChrome } from '@/store/interface-mode'
import { $chatOnboardingSolo } from '@/store/onboarding-intro'
import { $activeGatewayProfile } from '@/store/profile'
import { $freshDraftReady, $gatewayState } from '@/store/session'
import { $focusedSessionIsTile } from '@/store/session-focus'
import { $setupGuideOnboarding } from '@/store/shell-policy'
import { usePortalShell } from '@/store/shell-variant'
import { $statusbarVisible } from '@/store/statusbar-prefs'

import { ChatView } from '../chat'
import { ChatSidebar } from '../chat/sidebar'
import { PortalSidebar } from '../chat/sidebar/portal/portal-sidebar'
import { type PageFrameKind, WorkspacePageFallback } from '../overlays/page-header-skeleton'
import { ArtifactsView, CapabilitiesView, MessagingView } from '../page-views'
import { PortalGatewayMenuPanel } from '../redesign/portal-gateway-menu-panel'
import { TerminalPaneChrome } from '../right-sidebar/terminal/chrome'
import { contributedRoutes, isOverlayView, NEW_CHAT_ROUTE, ROUTES_AREA, sessionRoute } from '../routes'
import { useStatusSnapshot } from '../shell/hooks/use-status-snapshot'
import { useStatusbarItems } from '../shell/hooks/use-statusbar-items'
import { ModelMenuPanel } from '../shell/model-menu-panel'
import { ReasoningMenuPanel } from '../shell/reasoning-menu-panel'
import { StatusbarControls } from '../shell/statusbar-controls'

import { latestChatActions, latestSidebarActions } from './latest-actions'
import { setStatusbarItemGroup, useStatusbarContributions } from './panes'
import { KeptChatRoutes } from './route-cover'
import type { SidebarActions, WiringActions } from './types'

export function LegacySessionRedirect() {
  const { sessionId } = useParams()

  return <Navigate replace to={sessionId ? sessionRoute(sessionId) : NEW_CHAT_ROUTE} />
}

export const SidebarSurface = memo(function SidebarSurface({
  actions,
  currentView
}: {
  actions: SidebarActions
  currentView: ComponentProps<typeof ChatSidebar>['currentView']
}) {
  const latestActions = useMemo(() => latestSidebarActions(actions), [actions])

  const portal = usePortalShell()
  const Sidebar = portal ? PortalSidebar : ChatSidebar
  // Under a route overlay the redesign's sidebar is covered, and beside a page it is collapsed, but
  // never gone (route-cover.tsx, NavigationWithChatSidebar): it keeps the view it was showing, so moving
  // between pages and overlays never re-renders it.
  const focusedTile = useStore($focusedSessionIsTile)
  const covered = portal && (isOverlayView(currentView) || (currentView !== 'chat' && !focusedTile))
  const [shownView, setShownView] = useState(currentView)

  if (!covered && shownView !== currentView) {
    setShownView(currentView)
  }

  return <Sidebar currentView={covered ? shownView : currentView} {...latestActions} />
})

export const TerminalSurface = memo(function TerminalSurface() {
  return (
    <div className="relative flex h-full min-h-0 flex-col overflow-hidden bg-(--ui-terminal-surface-background)">
      <TerminalPaneChrome />
    </div>
  )
})

/** Owns the statusbar's own data hooks (status snapshot poll, contributed
 *  items) so its 15s refresh — and any statusbar-only churn — re-renders the
 *  bar alone, never the chat/sidebar/terminal. */
export const StatusbarSurface = memo(function StatusbarSurface({
  actions,
  agentsOpen,
  chatOpen,
  commandCenterOpen,
  onClose
}: {
  actions: WiringActions
  agentsOpen: boolean
  chatOpen: boolean
  commandCenterOpen: boolean
  onClose?: () => void
}) {
  const portal = usePortalShell()
  const { t } = useI18n()
  const advanced = useStore($showsAdvancedChrome)
  // The mode-resolved bar visibility: in Simple it is up exactly when the bar carries something
  // that must stay reachable (the free tier's Sign in, onboarding progress).
  const statusbarVisible = useStore($statusbarVisible)
  const activeConnectionId = useStore($activeConnectionId)
  const activeGatewayProfile = useStore($activeGatewayProfile)
  const gatewayState = useStore($gatewayState)
  const setupGuide = useStore($setupGuideOnboarding)
  // The classic first run's demo layout hides the bar: behaviour, so it follows the preference.
  const demoLayout = useStore($chatOnboardingSolo) && !setupGuide
  const freshDraftReady = useStore($freshDraftReady)
  const gatewayScope = `${activeConnectionId ?? ''}\0${activeGatewayProfile}`

  // Status describes the selected gateway/profile, not a focused chat on another owner.
  const requestStatus = useCallback(
    <T,>(method: string, params?: Record<string, unknown>) =>
      requestGatewayForAgent<T>(activeConnectionId, activeGatewayProfile, method, params),
    [activeConnectionId, activeGatewayProfile]
  )

  const { inferenceStatus, statusSnapshot } = useStatusSnapshot(gatewayState, requestStatus, gatewayScope)
  const extraLeftItems = useStatusbarContributions('left')
  const extraRightItems = useStatusbarContributions('right')

  const { leftStatusbarItems, statusbarItems } = useStatusbarItems({
    agentsOpen,
    changeSessionCwd: actions.changeSessionCwd,
    chatOpen,
    commandCenterOpen,
    extraLeftItems,
    extraRightItems,
    freshDraftReady,
    gatewayState,
    inferenceStatus,
    openAgents: actions.openAgents,
    openCommandCenterSection: actions.openCommandCenterSection,
    requestGateway: actions.requestGateway,
    statusSnapshot,
    toggleCommandCenter: actions.toggleCommandCenter
  })

  if (demoLayout) {
    return null
  }

  if (onClose) {
    return (
      <>
        <PortalGatewayMenuPanel
          gatewayState={gatewayState}
          inferenceStatus={inferenceStatus}
          key={gatewayScope}
          onClose={onClose}
          onOpenSystem={() => actions.openCommandCenterSection('system')}
          statusSnapshot={statusSnapshot}
        />
        {(advanced || statusbarVisible) && (
          <details className="px-3 pb-3">
            <summary className="cursor-pointer text-xs text-muted-foreground">
              {t.commandCenter.sections.system}
            </summary>
            <StatusbarControls
              items={statusbarItems.filter(item => item.id !== 'gateway-health')}
              leftItems={leftStatusbarItems}
              sidebar
            />
          </details>
        )}
      </>
    )
  }

  return <StatusbarControls items={statusbarItems} leftItems={leftStatusbarItems} sidebar={portal} />
})

/** The workspace pane: the real route table (chat + full-page views + plugin
 *  routes). Subscribes to the gateway instance/state and ROUTES_AREA itself;
 *  the voice cap arrives as a prop. ChatView subscribes to its own session
 *  atoms, so streaming never round-trips through the controller. */
export const ChatRoutesSurface = memo(function ChatRoutesSurface({
  actions,
  maxVoiceRecordingSeconds
}: {
  actions: WiringActions
  maxVoiceRecordingSeconds?: number
}) {
  const activeConnectionId = useStore($activeConnectionId)
  const activeGatewayProfile = useStore($activeGatewayProfile)
  const gateway = useStore($gateway)
  const gatewayState = useStore($gatewayState)
  const routeSnapshot = useContributions(ROUTES_AREA)
  const routeContributions = contributedRoutes(routeSnapshot)

  const modelMenuContent = useMemo(
    () =>
      gatewayState === 'open' ? (
        <ModelMenuPanel
          gateway={gateway || undefined}
          onFollowDefaultModel={actions.followDefaultModel}
          onSelectModel={actions.selectModel}
          ownerConnectionId={activeConnectionId || undefined}
          profile={activeGatewayProfile}
          requestGateway={actions.requestGateway}
        />
      ) : null,
    [actions, activeConnectionId, activeGatewayProfile, gateway, gatewayState]
  )

  const reasoningMenuContent = useMemo(
    () =>
      gatewayState === 'open' ? (
        <ReasoningMenuPanel
          gateway={gateway || undefined}
          onSelectModel={actions.selectModel}
          ownerConnectionId={activeConnectionId || undefined}
          profile={activeGatewayProfile}
          requestGateway={actions.requestGateway}
        />
      ) : null,
    [actions, activeConnectionId, activeGatewayProfile, gateway, gatewayState]
  )

  const chatActions = useMemo(() => latestChatActions(actions), [actions])
  const portal = usePortalShell()

  const chatView = (
    <ChatView
      gateway={gateway}
      maxVoiceRecordingSeconds={maxVoiceRecordingSeconds}
      modelMenuContent={modelMenuContent}
      modelOptionsOwnerConnectionId={activeConnectionId || undefined}
      modelOptionsProfile={activeGatewayProfile}
      reasoningMenuContent={reasoningMenuContent}
      requestModelOptionsForOwner={actions.requestGateway}
      {...chatActions}
    />
  )

  // FULL-PAGE views (not chat): a page is not a tab-able surface, so the zone's
  // tab strip stands down while one is showing. That is `paneChrome.headerVeto`
  // on the contribution, not a DOM marker — the `data-zone-no-header` attribute
  // that used to ride this wrapper gated a body double-click toggle that no
  // longer exists, and nothing has read it since.
  // A built-in page waits in its own frame; a contributed page's layout is its own business.
  const page = (view: ReactNode, frame?: PageFrameKind) => (
    <div className="contents">
      <Suspense fallback={frame && <WorkspacePageFallback frame={frame} />}>{view}</Suspense>
    </div>
  )

  // Everything the router reads as a chat location (appViewForPath), redirects included.
  const chatRoutes = (
    <>
      <Route element={chatView} index />
      <Route element={chatView} path=":sessionId" />
      <Route element={<Navigate replace to={NEW_CHAT_ROUTE} />} path="new" />
      <Route element={<LegacySessionRedirect />} path="sessions/:sessionId" />
      <Route element={<Navigate replace to={NEW_CHAT_ROUTE} />} path="*" />
    </>
  )

  const pageRoutes = (
    <>
      <Route
        element={page(<CapabilitiesView setStatusbarItemGroup={setStatusbarItemGroup} />, 'capabilities')}
        path="capabilities"
      />
      <Route
        element={page(<MessagingView setStatusbarItemGroup={setStatusbarItemGroup} />, 'messaging')}
        path="messaging"
      />
      <Route
        element={page(<ArtifactsView setStatusbarItemGroup={setStatusbarItemGroup} />, 'library')}
        path="artifacts"
      />
      <Route element={null} path="agents" />
      <Route element={null} path="command-center" />
      <Route element={null} path="cron" />
      <Route element={null} path="profiles" />
      <Route element={null} path="settings" />
      <Route element={null} path="starmap" />
      <Route element={null} path="webhooks" />
      {/* Registry-contributed pages (core features + plugins) render in the
            workspace pane like any built-in view — behind the same blast wall
            as every other contribution mount. */}
      {routeContributions.map(route => (
        <Route
          element={page(
            <ContribBoundary id={route.key}>
              <ContribRender render={route.render} />
            </ContribBoundary>
          )}
          key={route.key}
          path={route.path.slice(1)}
        />
      ))}
    </>
  )

  // The redesign paints pages over a kept chat (route-cover.tsx); classic swaps them in place.
  if (portal) {
    return <KeptChatRoutes chat={chatRoutes} pages={pageRoutes} />
  }

  return (
    <Routes>
      {chatRoutes}
      {pageRoutes}
    </Routes>
  )
})
