/**
 * Wiring surfaces — each pane is its own memoized component. Every surface
 * reads the reactive state it renders from at the leaf (its own atom
 * subscriptions) and reaches the controller's callbacks through the stable
 * `actions` bag, so a state change scoped to one surface (or a bare
 * wiring-controller tick) never re-renders another. This is what keeps the
 * layout tree's zones independently rendered — the whole point of the shell.
 */

import { useStore } from '@nanostores/react'
import { type ComponentProps, lazy, memo, type ReactNode, Suspense, useEffect, useMemo } from 'react'
import { Navigate, Route, Routes, useLocation, useNavigate, useParams } from 'react-router'

import { revealTreePane } from '@/components/pane-shell/tree/store'
import { ContribBoundary, ContribRender } from '@/contrib/react/boundary'
import { useContributions } from '@/contrib/react/use-contributions'
import { ESCAPE_PRIORITY, isTopEscapeLayer, pushEscapeLayer } from '@/lib/escape-layers'
import { triggerHaptic } from '@/lib/haptics'
import { isEditableTarget } from '@/lib/keybinds/combo'
import { $activeConnectionId } from '@/store/connections'
import { $gateway } from '@/store/gateway'
import { $chatOnboardingSolo } from '@/store/onboarding-intro'
import { $activeGatewayProfile } from '@/store/profile'
import { $freshDraftReady, $gatewayState } from '@/store/session'
import { goBackToPreviousView } from '@/store/view-history'

import { ChatView } from '../chat'
import { ChatSidebar } from '../chat/sidebar'
import { TerminalPaneChrome } from '../right-sidebar/terminal/chrome'
import { appViewForPath, contributedRoutes, isOverlayView, NEW_CHAT_ROUTE, ROUTES_AREA, sessionRoute } from '../routes'
import { useStatusSnapshot } from '../shell/hooks/use-status-snapshot'
import { useStatusbarItems } from '../shell/hooks/use-statusbar-items'
import { ModelMenuPanel } from '../shell/model-menu-panel'
import { ReasoningMenuPanel } from '../shell/reasoning-menu-panel'
import { StatusbarControls } from '../shell/statusbar-controls'

import { latestChatActions, latestSidebarActions } from './latest-actions'
import { setStatusbarItemGroup, useStatusbarContributions } from './panes'
import type { SidebarActions, WiringActions } from './types'

// Same lazy-view split as DesktopController — pages load on demand. The
// full-page views the workspace route table mounts live here; overlay views
// (agents/settings/…) are the controller's and stay in wiring.tsx.
const ArtifactsView = lazy(async () => ({ default: (await import('../artifacts')).ArtifactsView }))
const MessagingView = lazy(async () => ({ default: (await import('../messaging')).MessagingView }))
const CapabilitiesView = lazy(async () => ({ default: (await import('../capabilities')).CapabilitiesView }))

export function LegacySessionRedirect() {
  const { sessionId } = useParams()

  return <Navigate replace to={sessionId ? sessionRoute(sessionId) : NEW_CHAT_ROUTE} />
}

/**
 * Esc on a full page is the way back out of it — the same action, and the same
 * route history, as the titlebar's back control (`shell/titlebar-controls`).
 *
 * It has to be registered HERE, on the page's own layer, because a page is a
 * layer ABOVE the session view rather than a surface the chat was unmounted for:
 * the chat stays mounted and live underneath, so Esc must not reach it. The
 * composer stands down on a page (`$workspaceIsPage`, read by
 * `composerFocusBlockedBySurface`), and this layer is what Esc does instead.
 *
 * Priority is the lowest in the table — every other Esc layer (an overlay, the
 * zone editor, layout edit mode, a narrow-pane reveal, an in-flight drag) is
 * closer to the user than the page it covers, and its Esc must win over leaving
 * the page.
 */
function PageEscapeBack() {
  const navigate = useNavigate()

  useEffect(() => {
    const release = pushEscapeLayer(ESCAPE_PRIORITY.page)

    const onKeyDown = (event: KeyboardEvent) => {
      // A field on the page keeps its own Esc (clearing a search box, leaving an
      // input) — `defaultPrevented` covers a Radix dialog or a control that
      // already handled the key.
      if (
        event.key !== 'Escape' ||
        event.defaultPrevented ||
        isEditableTarget(event.target) ||
        !isTopEscapeLayer(ESCAPE_PRIORITY.page)
      ) {
        return
      }

      // Nothing behind (a cold start straight into a page): leave the key alone
      // rather than swallowing it.
      if (!goBackToPreviousView(navigate, revealTreePane)) {
        return
      }

      event.preventDefault()
      triggerHaptic('tap')
    }

    window.addEventListener('keydown', onKeyDown)

    return () => {
      window.removeEventListener('keydown', onKeyDown)
      release()
    }
  }, [navigate])

  return null
}

export const SidebarSurface = memo(function SidebarSurface({
  actions,
  currentView
}: {
  actions: SidebarActions
  currentView: ComponentProps<typeof ChatSidebar>['currentView']
}) {
  const latestActions = useMemo(() => latestSidebarActions(actions), [actions])

  return <ChatSidebar currentView={currentView} {...latestActions} />
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
  commandCenterOpen
}: {
  actions: WiringActions
  agentsOpen: boolean
  chatOpen: boolean
  commandCenterOpen: boolean
}) {
  const activeConnectionId = useStore($activeConnectionId)
  const activeGatewayProfile = useStore($activeGatewayProfile)
  const gatewayState = useStore($gatewayState)
  const demoLayout = useStore($chatOnboardingSolo)
  const freshDraftReady = useStore($freshDraftReady)
  const gatewayScope = `${activeConnectionId ?? ''}\0${activeGatewayProfile}`
  const { inferenceStatus, statusSnapshot } = useStatusSnapshot(gatewayState, actions.requestGateway, gatewayScope)
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

  // The demo layout (first-run intro) has no status bar.
  return demoLayout ? null : <StatusbarControls items={statusbarItems} leftItems={leftStatusbarItems} />
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
  const location = useLocation()
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

  // A full page is an OVERLAY above the session view, never a replacement for
  // it: the base layer is always the session view a chat route renders, so a
  // page keeps the chat MOUNTED beneath it (navigating to a page must not
  // unmount the chat) and a page can never outlive its route — any navigation
  // back to a session route drops the layer and shows that session. The layer
  // covers the same area the page used to fill, and paints the same surface, so
  // the result is visually indistinguishable from the old replace-the-workspace
  // behavior.
  //
  // `absolute inset-0` positions against the pane body (the zone's own
  // `relative` box), so the layer covers exactly the pane the page used to own.
  // The zone's tab strip still stands down while a page is showing — that is
  // `paneChrome.headerVeto` on the contribution, driven by `$workspaceIsPage`.
  //
  // `data-glass-raised` is load-bearing, not decoration: in window glass the
  // app thins `--ui-chat-surface-background` to `transparent` on purpose
  // (styles.css, `:root[data-hermes-glass]`), so a layer wearing only that token
  // paints NOTHING and the live session view underneath shows through the page.
  // The marker re-resolves the token to the near-opaque chrome mix for this
  // subtree — the same mechanism the overlay card uses for its raised surface.
  const page = (view: ReactNode) => (
    <div className="absolute inset-0 z-10 bg-(--ui-chat-surface-background)" data-glass-raised="">
      {/* Esc belongs to the page while it is the layer on top: it goes back out
          of it, and never reaches the live chat mounted beneath. */}
      <PageEscapeBack />
      <Suspense fallback={null}>{view}</Suspense>
    </div>
  )

  // An overlay route (settings/…) owns the window as a card over the shell, so
  // the workspace pane shows nothing under it — the session view stands only on
  // chat and page routes.
  const base = isOverlayView(appViewForPath(location.pathname)) ? null : chatView

  return (
    <>
      {base}
      <Routes>
        <Route element={page(<CapabilitiesView setStatusbarItemGroup={setStatusbarItemGroup} />)} path="capabilities" />
        <Route element={page(<MessagingView setStatusbarItemGroup={setStatusbarItemGroup} />)} path="messaging" />
        <Route element={page(<ArtifactsView setStatusbarItemGroup={setStatusbarItemGroup} />)} path="artifacts" />
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
        {/* Session, index, and overlay paths render no layer: the base session
            view already stands (or nothing, under an overlay card). Overlay
            paths fall through the `:sessionId` rung — they are reserved, so the
            session-id parser never mistakes them for a session. */}
        <Route element={null} index />
        <Route element={null} path=":sessionId" />
        <Route element={<Navigate replace to={NEW_CHAT_ROUTE} />} path="new" />
        <Route element={<LegacySessionRedirect />} path="sessions/:sessionId" />
        <Route element={<Navigate replace to={NEW_CHAT_ROUTE} />} path="*" />
      </Routes>
    </>
  )
})
