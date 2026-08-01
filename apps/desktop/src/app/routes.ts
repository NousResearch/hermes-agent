import { atom } from 'nanostores'
import type { ReactNode } from 'react'

import { noteActiveTreeGroup, revealTreePane } from '@/components/pane-shell/tree/store'
import { registry } from '@/contrib/registry'

type NavigateLike = (to: string, options?: { replace?: boolean }) => void

export const SESSION_ROUTE_PREFIX = '/'
export const NEW_CHAT_ROUTE = '/'
export const SETTINGS_ROUTE = '/settings'
export const COMMAND_CENTER_ROUTE = '/command-center'
export const SKILLS_ROUTE = '/skills'
export const MESSAGING_ROUTE = '/messaging'
export const WEBHOOKS_ROUTE = '/webhooks'
export const ARTIFACTS_ROUTE = '/artifacts'
export const CRON_ROUTE = '/cron'
export const PROFILES_ROUTE = '/profiles'
export const AGENTS_ROUTE = '/agents'
export const STARMAP_ROUTE = '/starmap'

export type AppView =
  | 'agents'
  | 'artifacts'
  | 'chat'
  | 'command-center'
  | 'cron'
  // A contributed (plugin) full page at its own route — NOT chat. Without this
  // distinction contributed paths fell through appViewForPath's 'chat' default,
  // so the sidebar kept a session highlighted and the titlebar kept the
  // session-title dropdown while a plugin page was showing.
  | 'extension'
  | 'messaging'
  | 'profiles'
  | 'settings'
  | 'skills'
  | 'starmap'
  | 'webhooks'

export type AppRouteId =
  | 'agents'
  | 'artifacts'
  | 'command-center'
  | 'cron'
  | 'messaging'
  | 'new'
  | 'profiles'
  | 'settings'
  | 'skills'
  | 'starmap'
  | 'webhooks'

export interface AppRoute {
  id: AppRouteId
  path: string
  view: AppView
}

export const APP_ROUTES = [
  { id: 'new', path: NEW_CHAT_ROUTE, view: 'chat' },
  { id: 'settings', path: SETTINGS_ROUTE, view: 'settings' },
  { id: 'command-center', path: COMMAND_CENTER_ROUTE, view: 'command-center' },
  { id: 'skills', path: SKILLS_ROUTE, view: 'skills' },
  { id: 'messaging', path: MESSAGING_ROUTE, view: 'messaging' },
  { id: 'webhooks', path: WEBHOOKS_ROUTE, view: 'webhooks' },
  { id: 'artifacts', path: ARTIFACTS_ROUTE, view: 'artifacts' },
  { id: 'cron', path: CRON_ROUTE, view: 'cron' },
  { id: 'profiles', path: PROFILES_ROUTE, view: 'profiles' },
  { id: 'agents', path: AGENTS_ROUTE, view: 'agents' },
  { id: 'starmap', path: STARMAP_ROUTE, view: 'starmap' }
] as const satisfies readonly AppRoute[]

const APP_VIEW_BY_PATH = new Map<string, AppView>(APP_ROUTES.map(route => [route.path, route.view]))
const RESERVED_PATHS: ReadonlySet<string> = new Set(APP_ROUTES.map(route => route.path))

// ── Contributed routes — the `routes` registry area ─────────────────────────
// A contribution mounts a FULL PAGE in the workspace pane at `data.path`
// (`render` on the contribution itself, like every other area). Contributed
// paths are reserved exactly like APP_ROUTES so the session-id parser never
// mistakes them for a session route. Navigate with `host.navigate(path)`.

export const ROUTES_AREA = 'routes'

/** Payload of a `routes` contribution's `data`. */
export interface RouteContribution {
  /** Absolute path, e.g. `/kanban`. One segment; no params. */
  path: string
}

export function contributedRoutes(): Array<{ key: string; path: string; title?: string; render: () => ReactNode }> {
  return registry
    .getArea(ROUTES_AREA)
    .map(c => ({
      key: `${c.source ?? 'core'}:${c.id}`,
      path: (c.data as RouteContribution | undefined)?.path ?? '',
      title: c.title,
      render: c.render!
    }))
    .filter(route => Boolean(route.path.startsWith('/') && route.render) && !RESERVED_PATHS.has(route.path))
}

function isContributedPath(pathname: string): boolean {
  return contributedRoutes().some(route => route.path === pathname)
}

// ── Contributed sidebar nav — the `sidebar.nav` registry area ────────────────
// A DATA contribution adds a row to the sidebar's top nav (below Artifacts).
// Pair with a ROUTES_AREA page: the row navigates to `path` and lights up
// while the app is there.

export const SIDEBAR_NAV_AREA = 'sidebar.nav'

/** Payload of a `sidebar.nav` data contribution. */
export interface SidebarNavContribution {
  /** Codicon name, e.g. `'project'`. */
  codicon: string
  label: string
  /** Route to navigate to (usually a contributed page's path). */
  path: string
}

// Views that render as a full-screen modal card (OverlayView) over the shell.
// While one is open the app's titlebar control clusters must hide so they don't
// bleed over the overlay (they sit at a higher z-index than the overlay card).
export const OVERLAY_VIEWS: ReadonlySet<AppView> = new Set([
  'agents',
  'command-center',
  'cron',
  'profiles',
  'settings',
  'starmap',
  'webhooks'
])

export function isOverlayView(view: AppView): boolean {
  return OVERLAY_VIEWS.has(view)
}

/** The pathname of a router target. Every classifier below reasons about a
 *  PATH, but callers navigate to full targets (`/skills?tab=mcp`), and an
 *  unstripped query reaches the session-id parser — `/skills?tab=mcp` reads as
 *  the session `skills?tab=mcp`, so Capabilities classifies as a chat.
 *  `sessionRoute` percent-encodes ids, so `?`/`#` can only start a query or a
 *  hash. */
export function routePathname(to: string): string {
  const cut = to.search(/[?#]/)

  return cut === -1 ? to : to.slice(0, cut)
}

export function isNewChatRoute(pathname: string): boolean {
  return routePathname(pathname) === NEW_CHAT_ROUTE
}

export function routeSessionId(pathname: string): string | null {
  const path = routePathname(pathname)

  if (!path.startsWith(SESSION_ROUTE_PREFIX) || RESERVED_PATHS.has(path) || isContributedPath(path)) {
    return null
  }

  const id = path.slice(SESSION_ROUTE_PREFIX.length)

  return id && !id.includes('/') ? decodeURIComponent(id) : null
}

/**
 * The primary composer's durable scope key candidate: the route is the source
 * of truth for which chat is on screen, so prefer its (stable) stored session
 * id over a store selection that can be momentarily null/stale mid-switch
 * (#59305). A genuine new-chat route always wins with `null`, never falling
 * back to a leftover selection from the chat just left. A non-chat route
 * (settings, an overlay) has no session opinion, so the store selection passes
 * through unchanged.
 */
export function primaryRouteSelectedSessionId(pathname: string, storeSelectedSessionId: string | null): string | null {
  if (isNewChatRoute(pathname)) {
    return null
  }

  return routeSessionId(pathname) ?? storeSelectedSessionId
}

export function sessionRoute(sessionId: string): string {
  return `${SESSION_ROUTE_PREFIX}${encodeURIComponent(sessionId)}`
}

export function appViewForPath(pathname: string): AppView {
  const path = routePathname(pathname)

  if (isNewChatRoute(path) || routeSessionId(path)) {
    return 'chat'
  }

  if (isContributedPath(path)) {
    return 'extension'
  }

  return APP_VIEW_BY_PATH.get(path) ?? 'chat'
}

/** Does `to` land on a full page rendered INSIDE the workspace pane
 *  (skills/messaging/artifacts/contributed routes)? Overlays don't count —
 *  they float over whatever the workspace is already showing. */
function isWorkspacePageRoute(to: string): boolean {
  const view = appViewForPath(to)

  return view !== 'chat' && !isOverlayView(view)
}

/** True while the workspace pane shows a FULL PAGE (skills/messaging/
 *  artifacts/plugin routes) instead of the chat. Published by the wiring
 *  (which owns the router location); the workspace pane contribution mirrors
 *  it as `headerVeto` so the zone tab bar stands down on pages. Overlays
 *  (settings/…) don't count — the chat stays beneath them. */
export const $workspaceIsPage = atom(false)

function revealWorkspacePane(): void {
  noteActiveTreeGroup(null)
  revealTreePane('workspace')
}

/**
 * Point the workspace at `pathname`: mirror "showing a full page" into
 * `$workspaceIsPage`, and FRONT the pane when it is one.
 *
 * A page renders inside `workspace`, so a main zone parked on a session tile
 * keeps the tile on screen while the route and the page content change behind
 * it — the navigation looks dead (#72602). Session switches already front the
 * pane in `store/session-states.ts`; pages had no equivalent.
 *
 * The router location drives this, so every entry point gets it without opting
 * in: sidebar, keybinds, command palette, Command Center, contributed
 * statusbar/titlebar `to` targets, back/forward, and cold-start restore.
 */
export function syncWorkspaceRoute(pathname: string): void {
  const isPage = isWorkspacePageRoute(pathname)

  if (isPage !== $workspaceIsPage.get()) {
    $workspaceIsPage.set(isPage)
  }

  if (isPage) {
    revealWorkspacePane()
  }
}

/**
 * Navigate to `to`, fronting the workspace pane when it is a page route.
 *
 * `syncWorkspaceRoute` covers route CHANGES; this covers the RE-CLICK, the one
 * case it can't see — hitting Capabilities while already on `/skills` with a
 * tile focused leaves the location untouched, so no effect fires and only an
 * imperative reveal brings the page back. Use it wherever a nav affordance can
 * be triggered from the page it targets.
 */
export function navigateToWorkspacePage(navigate: NavigateLike, to: string, options?: { replace?: boolean }): void {
  navigate(to, options)

  if (isWorkspacePageRoute(to)) {
    revealWorkspacePane()
  }
}
