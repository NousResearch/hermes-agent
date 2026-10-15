import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// A page route renders INSIDE the workspace pane, so opening one fronts the
// workspace pane — synchronously, from the nav affordance itself
// (`routes.navigateToWorkspacePage` → `revealWorkspacePane`), in the same tick
// as the click. A stamp taken when the route LANDS therefore reads `workspace`
// and erases the session tile the user was reading, so Back landed on the first
// session tab: the workspace. The tab is recorded where it is FRONTED — every
// layout commit — plus once when a view is entered, for a restored tab fronted
// before the route was known.
//
// The routes below are the app's own, in the app's own order: nav affordance,
// then the wiring effect that records the route.

describe('Back restores the tab the view fronted', () => {
  beforeEach(() => {
    window.localStorage.clear()
    vi.resetModules()
  })

  afterEach(() => {
    vi.resetModules()
  })

  async function setup() {
    const tree = await import('@/components/pane-shell/tree/store')
    const model = await import('@/components/pane-shell/tree/model')
    const history = await import('@/store/view-history')
    const routes = await import('@/app/routes')
    const { registry } = await import('@/contrib/registry')

    for (const id of ['workspace', 'session-tile:a']) {
      registry.register({
        area: 'panes',
        data: id === 'workspace' ? { placement: 'main', uncloseable: true } : { placement: 'main' },
        id,
        render: () => null,
        title: id
      })
    }

    // The user's own shape: the session tile stacks with the workspace.
    tree.declareDefaultTree(model.group(['workspace', 'session-tile:a'], { active: 'workspace', id: 'grp-main' }))

    // `app/contrib/wiring.tsx`, verbatim: sync, record, then record the tab the
    // entered view has.
    const recordRoute = (pathname: string) => {
      routes.syncWorkspaceRoute(pathname)
      history.noteViewChange(pathname)
      const fronted = tree.activeMainStripPane()

      if (fronted) {
        history.noteActiveViewPane(fronted)
      }
    }

    return { history, recordRoute, routes, tree }
  }

  it('returns to a session tile a page displaced, not to the workspace', async () => {
    const { history, recordRoute, routes, tree } = await setup()

    recordRoute('/session-1')
    // The user fronts their session tile: no navigation, so only the tree writes it.
    tree.activateTreePane('grp-main', 'session-tile:a')

    // Opening a page: `navigate()` then the fronting, before any effect runs.
    routes.navigateToWorkspacePage(vi.fn(), '/capabilities')
    recordRoute('/capabilities')

    expect(history.viewPaneFor('/session-1')).toBe('session-tile:a')
    expect(history.goBackToPreviousView(vi.fn(), tree.revealTreePane)).toBe(true)
    expect(tree.activeMainStripPane()).toBe('session-tile:a')
  })

  it('returns to a tile opened beside the chat by a reveal, not only a tab click', async () => {
    const { history, recordRoute, routes, tree } = await setup()

    recordRoute('/session-1')
    // "Open as tile" / a drag from the sidebar fronts it through `revealTreePane`.
    tree.revealTreePane('session-tile:a')

    routes.navigateToWorkspacePage(vi.fn(), '/artifacts')
    recordRoute('/artifacts')

    expect(history.viewPaneFor('/session-1')).toBe('session-tile:a')
    expect(history.goBackToPreviousView(vi.fn(), tree.revealTreePane)).toBe(true)
    expect(tree.activeMainStripPane()).toBe('session-tile:a')
  })

  it('keeps a tab a RESTORED layout fronted before the route was known', async () => {
    const { history, recordRoute, routes, tree } = await setup()

    // The saved layout comes back with the tile fronted, and no route is
    // recorded yet — so nothing in the tree listener had anywhere to write it.
    tree.revealTreePane('session-tile:a')
    expect(history.$viewHistory.get().entries).toHaveLength(0)

    recordRoute('/session-1')
    routes.navigateToWorkspacePage(vi.fn(), '/capabilities')
    recordRoute('/capabilities')

    expect(history.viewPaneFor('/session-1')).toBe('session-tile:a')
    expect(history.goBackToPreviousView(vi.fn(), tree.revealTreePane)).toBe(true)
    expect(tree.activeMainStripPane()).toBe('session-tile:a')
  })

  it('records the workspace as a page view\u2019s own tab, since the page renders there', async () => {
    const { history, recordRoute, routes, tree } = await setup()

    recordRoute('/session-1')
    tree.activateTreePane('grp-main', 'session-tile:a')
    routes.navigateToWorkspacePage(vi.fn(), '/capabilities')
    recordRoute('/capabilities')

    expect(history.viewPaneFor('/capabilities')).toBe('workspace')
  })

  it('remembers the workspace when that is the tab the user fronted', async () => {
    const { history, recordRoute, routes, tree } = await setup()

    recordRoute('/session-1')
    tree.activateTreePane('grp-main', 'session-tile:a')
    tree.activateTreePane('grp-main', 'workspace')

    routes.navigateToWorkspacePage(vi.fn(), '/capabilities')
    recordRoute('/capabilities')

    expect(history.viewPaneFor('/session-1')).toBe('workspace')
  })
})
