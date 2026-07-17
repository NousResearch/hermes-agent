import { beforeEach, describe, expect, it, vi } from 'vitest'

/**
 * Regression harness for #65985 — the outer Terminal/Logs pane-tab context-menu
 * "Close" only minimized the zone (tab stayed visible); only the undiscoverable
 * middle-click actually removed the tab.
 *
 * Root cause: the context menu called `closeTreePane`, which for a TOOL PANEL
 * (terminal/logs — a collapse pane bound via `bindPaneCollapse`) runs the
 * registered closer, and that closer only toggles the pane's open store off
 * (collapse to a rail). Middle-click / the tab ✕ instead routed collapse panes
 * to `dismissTreePane`, which removes them from the layout.
 *
 * The fix funnels BOTH gestures through one store route, `closePaneTab`, so the
 * context-menu "Close" removes the tab exactly like middle-click. We assert that
 * routing at the store level (the cleanest layer — it is a pure store action).
 *
 * Fresh module state per test (localStorage + resetModules) because
 * `markCollapsePane` / `registerPaneCloser` mutate module-level maps.
 */
async function loadTree() {
  const tree = await import('@/components/pane-shell/tree/store')
  const model = await import('@/components/pane-shell/tree/model')

  const paneIds = () => model.allPaneIds(tree.$layoutTree.get()!)

  return { model, paneIds, tree }
}

describe('#65985 closePaneTab — context-menu Close removes tool-panel tabs', () => {
  beforeEach(() => {
    window.localStorage.clear()
    vi.resetModules()
  })

  it('REMOVES a terminal/logs collapse pane from the layout, without running its collapse closer', async () => {
    const { model, paneIds, tree } = await loadTree()

    // Mirror bindPaneCollapse('terminal', …): a collapse pane whose registered
    // closer only MINIMIZES (toggles its open store off) — it never removes the
    // tab. This is exactly the closer that made the old context menu no-op.
    const collapse = vi.fn()
    tree.markCollapsePane('terminal')
    tree.registerPaneCloser('terminal', collapse)

    tree.declareDefaultTree(model.split('row', [model.group(['workspace']), model.group(['terminal'])], [3, 1]))
    expect(paneIds()).toContain('terminal')

    // The context-menu "Close" route (same one middle-click / the ✕ now use).
    tree.closePaneTab('terminal')

    // Tab is GONE from the layout — not merely collapsed…
    expect(paneIds()).not.toContain('terminal')
    // …and it did NOT take the minimize path (the collapse closer never ran).
    expect(collapse).not.toHaveBeenCalled()
  })

  it('matches the middle-click removal path (dismissTreePane) for a collapse pane', async () => {
    const { model, paneIds, tree } = await loadTree()
    tree.markCollapsePane('logs')
    tree.registerPaneCloser('logs', vi.fn())

    tree.declareDefaultTree(model.split('row', [model.group(['workspace']), model.group(['logs'])], [3, 1]))

    // closePaneTab (context menu) and dismissTreePane (the real middle-click
    // removal path) leave the SAME surviving panes.
    tree.closePaneTab('logs')
    const afterContextMenuClose = paneIds()

    tree.declareDefaultTree(model.split('row', [model.group(['workspace']), model.group(['logs'])], [3, 1]))
    tree.dismissTreePane('logs')
    const afterMiddleClick = paneIds()

    expect(afterContextMenuClose).toEqual(afterMiddleClick)
    expect(afterContextMenuClose).not.toContain('logs')
  })

  it('leaves OTHER tab kinds (a session tile / store-bound pane) on their Close route — closer runs, tab stays', async () => {
    const { model, paneIds, tree } = await loadTree()

    // A non-collapse pane with a registered closer (e.g. a session tile / a
    // store-bound pane). closePaneTab must route it through closeTreePane, whose
    // closer OWNS the close — the pane is NOT dismissed from the tree here.
    const closeSession = vi.fn()
    tree.registerPaneCloser('session-tile:abc', closeSession)

    tree.declareDefaultTree(
      model.split('row', [model.group(['workspace']), model.group(['session-tile:abc'])], [3, 1])
    )

    tree.closePaneTab('session-tile:abc')

    expect(closeSession).toHaveBeenCalledTimes(1)
    // Behavior for this tab kind is unchanged — its closer decides, the tree
    // still holds the pane (the fix is scoped to collapse panes only).
    expect(paneIds()).toContain('session-tile:abc')
  })
})
