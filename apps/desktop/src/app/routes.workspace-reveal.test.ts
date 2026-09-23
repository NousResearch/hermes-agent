/**
 * A full page (Capabilities/Messaging/Artifacts/a contributed route) renders
 * INSIDE the `workspace` pane, so navigating to one has to front that pane —
 * otherwise a main zone parked on a session tile keeps the tile on screen and
 * the click looks dead until the app restarts (#72602).
 *
 * Two layers, both covered here: `syncWorkspaceRoute` (the router location,
 * every entry point) and `navigateToWorkspacePage` (the re-click, where the
 * location doesn't change).
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { registry } from '@/contrib/registry'
import { host } from '@/sdk'

import {
  $workspaceIsPage,
  AGENTS_ROUTE,
  appViewForPath,
  ARTIFACTS_ROUTE,
  CAPABILITIES_ROUTE,
  CRON_ROUTE,
  MESSAGING_ROUTE,
  navigateToWorkspacePage,
  NEW_CHAT_ROUTE,
  routePathname,
  ROUTES_AREA,
  routeSessionId,
  sessionRoute,
  SETTINGS_ROUTE,
  syncWorkspaceRoute
} from './routes'

vi.mock('@/components/pane-shell/tree/store', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  noteActiveTreeGroup: vi.fn(),
  revealTreePane: vi.fn()
}))

const { noteActiveTreeGroup, revealTreePane } = await import('@/components/pane-shell/tree/store')

const CONTRIBUTED_ROUTE = '/kanban'

function contributeRoute(): () => void {
  return registry.register({
    area: ROUTES_AREA,
    data: { path: CONTRIBUTED_ROUTE },
    id: 'test-route',
    render: () => null
  })
}

/** Did the workspace pane get fronted? Both calls, or the tab stays put. */
const fronted = () =>
  vi.mocked(revealTreePane).mock.calls.some(([pane]) => pane === 'workspace') &&
  vi.mocked(noteActiveTreeGroup).mock.calls.some(([group]) => group === null)

beforeEach(() => {
  vi.mocked(revealTreePane).mockClear()
  vi.mocked(noteActiveTreeGroup).mockClear()
  $workspaceIsPage.set(false)
})

afterEach(() => {
  $workspaceIsPage.set(false)
})

describe('routePathname', () => {
  it('keeps a bare path and drops a query or hash', () => {
    expect(routePathname(CAPABILITIES_ROUTE)).toBe('/capabilities')
    expect(routePathname('/capabilities?tab=connectors')).toBe('/capabilities')
    expect(routePathname('/capabilities?tab=connectors&server=ctx7')).toBe('/capabilities')
    expect(routePathname('/settings#keys')).toBe('/settings')
  })

  it('leaves an encoded session id alone', () => {
    const route = sessionRoute('a?b#c')

    expect(routePathname(route)).toBe(route)
    expect(routeSessionId(route)).toBe('a?b#c')
  })
})

describe('classification of targets carrying a query', () => {
  // The palette navigates to every one of these (Capabilities tabs, MCP
  // servers), and Settings redirects old /settings?tab=mcp deep links to the
  // last one. Unstripped, they parsed as SESSION ids and read as 'chat'.
  it.each([
    [`${CAPABILITIES_ROUTE}?tab=skills`, 'capabilities'],
    [`${CAPABILITIES_ROUTE}?tab=toolsets`, 'capabilities'],
    [`${CAPABILITIES_ROUTE}?tab=connectors&server=ctx7`, 'capabilities'],
    [`${SETTINGS_ROUTE}?tab=keys`, 'settings']
  ])('%s is not a session route', (to, view) => {
    expect(routeSessionId(to)).toBeNull()
    expect(appViewForPath(to)).toBe(view)
  })
})

describe('syncWorkspaceRoute', () => {
  it('publishes and fronts on a page route', () => {
    syncWorkspaceRoute(CAPABILITIES_ROUTE)

    expect($workspaceIsPage.get()).toBe(true)
    expect(fronted()).toBe(true)
  })

  it('fronts on a page route reached with a query', () => {
    syncWorkspaceRoute(`${CAPABILITIES_ROUTE}?tab=connectors`)

    expect($workspaceIsPage.get()).toBe(true)
    expect(fronted()).toBe(true)
  })

  it('fronts when moving between two pages — the atom never changes, the tab must', () => {
    syncWorkspaceRoute(ARTIFACTS_ROUTE)
    vi.mocked(revealTreePane).mockClear()
    vi.mocked(noteActiveTreeGroup).mockClear()

    syncWorkspaceRoute(MESSAGING_ROUTE)

    expect($workspaceIsPage.get()).toBe(true)
    expect(fronted()).toBe(true)
  })

  it('fronts on a contributed page route', () => {
    const dispose = contributeRoute()

    try {
      syncWorkspaceRoute(CONTRIBUTED_ROUTE)

      expect(appViewForPath(CONTRIBUTED_ROUTE)).toBe('extension')
      expect(fronted()).toBe(true)
    } finally {
      dispose()
    }
  })

  it.each([
    ['a session route', sessionRoute('sess-a')],
    ['the new-chat route', NEW_CHAT_ROUTE],
    ['an overlay', SETTINGS_ROUTE],
    ['an overlay with a query', `${SETTINGS_ROUTE}?tab=keys`],
    ['another overlay', CRON_ROUTE],
    ['yet another overlay', AGENTS_ROUTE]
  ])('leaves the tab alone on %s', (_label, to) => {
    syncWorkspaceRoute(to)

    expect($workspaceIsPage.get()).toBe(false)
    expect(revealTreePane).not.toHaveBeenCalled()
  })
})

describe('navigateToWorkspacePage', () => {
  it('navigates and fronts, so a re-click on the page you are already on still shows it', () => {
    const navigate = vi.fn()

    navigateToWorkspacePage(navigate, CAPABILITIES_ROUTE)

    expect(navigate).toHaveBeenCalledWith(CAPABILITIES_ROUTE, undefined)
    expect(fronted()).toBe(true)
  })

  it.each([
    `${CAPABILITIES_ROUTE}?tab=skills`,
    `${CAPABILITIES_ROUTE}?tab=toolsets`,
    `${CAPABILITIES_ROUTE}?tab=connectors&server=ctx7`
  ])('fronts for the palette target %s', to => {
    navigateToWorkspacePage(vi.fn(), to)

    expect(fronted()).toBe(true)
  })

  it('passes navigation options through', () => {
    const navigate = vi.fn()

    navigateToWorkspacePage(navigate, ARTIFACTS_ROUTE, { replace: true })

    expect(navigate).toHaveBeenCalledWith(ARTIFACTS_ROUTE, { replace: true })
  })

  it('navigates without fronting for chat and overlay targets', () => {
    const navigate = vi.fn()

    navigateToWorkspacePage(navigate, sessionRoute('sess-a'))
    navigateToWorkspacePage(navigate, SETTINGS_ROUTE)

    expect(navigate).toHaveBeenCalledTimes(2)
    expect(revealTreePane).not.toHaveBeenCalled()
  })
})

/**
 * `host.navigate` is the only nav door a plugin has (Kanban's ⌘K row, statusbar
 * count, ⌘⌥N). It writes the hash, which the router follows on a CHANGE — but
 * re-issuing the current route fires nothing, so it must reveal imperatively
 * like the sidebar does.
 */
describe('host.navigate', () => {
  it('fronts the workspace pane even when already on the contributed page', () => {
    const dispose = contributeRoute()

    try {
      window.location.hash = `#${CONTRIBUTED_ROUTE}`
      vi.mocked(revealTreePane).mockClear()
      vi.mocked(noteActiveTreeGroup).mockClear()

      host.navigate(CONTRIBUTED_ROUTE)

      expect(window.location.hash).toBe(`#${CONTRIBUTED_ROUTE}`)
      expect(fronted()).toBe(true)
    } finally {
      dispose()
    }
  })

  it('does not front the pane for a chat route', () => {
    host.navigate(sessionRoute('sess-a'))

    expect(revealTreePane).not.toHaveBeenCalled()
  })
})
