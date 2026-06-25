export const SESSION_ROUTE_PREFIX = '/'
export const NEW_CHAT_ROUTE = '/'
export const SETTINGS_ROUTE = '/settings'
export const COMMAND_CENTER_ROUTE = '/command-center'
export const SKILLS_ROUTE = '/skills'
export const MESSAGING_ROUTE = '/messaging'
export const ARTIFACTS_ROUTE = '/artifacts'
export const CRON_ROUTE = '/cron'
export const PROFILES_ROUTE = '/profiles'
export const AGENTS_ROUTE = '/agents'

export type AppView =
  | 'agents'
  | 'artifacts'
  | 'chat'
  | 'command-center'
  | 'cron'
  | 'messaging'
  | 'plugin'
  | 'profiles'
  | 'settings'
  | 'skills'

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
  { id: 'artifacts', path: ARTIFACTS_ROUTE, view: 'artifacts' },
  { id: 'cron', path: CRON_ROUTE, view: 'cron' },
  { id: 'profiles', path: PROFILES_ROUTE, view: 'profiles' },
  { id: 'agents', path: AGENTS_ROUTE, view: 'agents' }
] as const satisfies readonly AppRoute[]

const APP_VIEW_BY_PATH = new Map<string, AppView>(APP_ROUTES.map(route => [route.path, route.view]))
const RESERVED_PATHS: ReadonlySet<string> = new Set(APP_ROUTES.map(route => route.path))

// Views that render as a full-screen modal card (OverlayView) over the shell.
// While one is open the app's titlebar control clusters must hide so they don't
// bleed over the overlay (they sit at a higher z-index than the overlay card).
export const OVERLAY_VIEWS: ReadonlySet<AppView> = new Set(['agents', 'command-center', 'cron', 'profiles', 'settings'])

export function isOverlayView(view: AppView): boolean {
  return OVERLAY_VIEWS.has(view)
}

export function isNewChatRoute(pathname: string): boolean {
  return pathname === NEW_CHAT_ROUTE
}

export function routeSessionId(pathname: string, extraReservedPaths?: ReadonlySet<string>): string | null {
  if (
    !pathname.startsWith(SESSION_ROUTE_PREFIX) ||
    RESERVED_PATHS.has(pathname) ||
    Boolean(extraReservedPaths?.has(pathname))
  ) {
    return null
  }

  const id = pathname.slice(SESSION_ROUTE_PREFIX.length)

  return id && !id.includes('/') ? decodeURIComponent(id) : null
}

export function sessionRoute(sessionId: string): string {
  return `${SESSION_ROUTE_PREFIX}${encodeURIComponent(sessionId)}`
}

export function appViewForPath(pathname: string, extraReservedPaths?: ReadonlySet<string>): AppView {
  if (extraReservedPaths?.has(pathname)) {
    return 'plugin'
  }

  if (isNewChatRoute(pathname) || routeSessionId(pathname, extraReservedPaths)) {
    return 'chat'
  }

  return APP_VIEW_BY_PATH.get(pathname) ?? 'chat'
}

// Dashboard plugin icons are declared as web/lucide names (e.g. "Package") in
// the manifest, but the desktop sidebar renders the codicon font. Map the names
// kanban + the bundled plugins use to codicon glyphs; unknown names fall back to
// a neutral "extensions" puzzle-piece glyph so a new plugin still gets an icon.
const PLUGIN_ICON_CODICON: Record<string, string> = {
  Package: 'package',
  LayoutGrid: 'layout',
  Kanban: 'project',
  Columns: 'layout',
  Boxes: 'package',
  Box: 'package',
  Activity: 'pulse',
  Zap: 'zap',
  Heart: 'heart',
  Star: 'star-full',
  Code: 'code',
  Eye: 'eye',
  Puzzle: 'extensions'
}

/** Resolve a plugin manifest icon name to a codicon glyph for the sidebar. */
export function pluginIconCodicon(name: string | undefined): string {
  return (name && PLUGIN_ICON_CODICON[name]) || 'extensions'
}
