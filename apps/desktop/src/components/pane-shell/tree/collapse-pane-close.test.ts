import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { atom } from 'nanostores'

import { allPaneIds } from './model'

// Repro for "the logs tab ✕ kills the pane and the toggle can't bring it
// back": a tool panel (terminal/logs) bound via bindPaneCollapse had its tab ✕
// routed through dismissTreePane, which removed the pane from the layout but
// never synced the owning store — so the ⌘K "Toggle logs" toggle was stale
// (nanostores don't fire on a same-value .set), and its open listener called
// setPaneCollapsed, which is a no-op when the pane isn't in the tree. The tab
// was gone with no way back short of a layout reset.
//
// The fix routes the tab ✕ through closeCollapsePane (dismiss + store sync),
// and bindPaneCollapse's open listener calls revealTreePane (un-dismiss +
// re-adopt) instead of setPaneCollapsed.

describe('collapse pane tab close + toggle recovery', () => {
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
    const { registry } = await import('@/contrib/registry')

    // A tool panel (logs): placement 'bottom', not uncloseable.
    registry.register({
      id: 'logs',
      area: 'panes',
      title: 'logs',
      data: { placement: 'bottom' },
      render: () => null
    })
    // The main workspace, so adoption has an anchor.
    registry.register({
      id: 'workspace',
      area: 'panes',
      title: 'workspace',
      data: { placement: 'main', uncloseable: true },
      render: () => null
    })

    tree.declareDefaultTree(
      model.split('column', [
        model.group(['workspace'], { active: 'workspace', id: 'grp-main' }),
        model.group(['logs'], { active: 'logs', id: 'grp-logs' })
      ])
    )

    // Mirror controller.tsx bindPaneCollapse: the owning store drives
    // visibility, the closer syncs it on ✕, the opener reveals.
    const $logsOpen = atom(true)

    // bindPaneCollapse inline (can't import the controller's private fn).
    tree.markCollapsePane('logs')
    tree.setPaneCollapsed('logs', !$logsOpen.get())
    $logsOpen.subscribe(isOpen => {
      if (isOpen) {
        tree.revealTreePane('logs')
      } else {
        tree.setPaneCollapsed('logs', true)
      }
    })
    tree.registerPaneCloser('logs', () => $logsOpen.set(false))
    tree.registerPaneOpener('logs', () => $logsOpen.set(true))

    return { tree, $logsOpen }
  }

  it('tab ✕ dismisses the pane AND syncs the owning store', async () => {
    const { tree, $logsOpen } = await setup()

    expect($logsOpen.get()).toBe(true)
    expect(allPaneIds(tree.$layoutTree.get()!)).toContain('logs')

    // The tab ✕ on a collapse pane.
    tree.closeCollapsePane('logs')

    // Pane is gone from the layout…
    expect(allPaneIds(tree.$layoutTree.get()!)).not.toContain('logs')
    // …and the store was synced to false.
    expect($logsOpen.get()).toBe(false)
  })

  it('the toggle brings the pane back after the tab ✕ closed it', async () => {
    const { tree, $logsOpen } = await setup()

    // Close via tab ✕.
    tree.closeCollapsePane('logs')
    expect($logsOpen.get()).toBe(false)
    expect(allPaneIds(tree.$layoutTree.get()!)).not.toContain('logs')

    // Re-open via the toggle (the ⌘K row / opener).
    $logsOpen.set(true)

    // The pane is back in the layout.
    expect(allPaneIds(tree.$layoutTree.get()!)).toContain('logs')
    expect($logsOpen.get()).toBe(true)
  })

  it('closeTabPane routes collapse panes through closeCollapsePane', async () => {
    const { tree, $logsOpen } = await setup()

    tree.closeTabPane('logs')

    expect(allPaneIds(tree.$layoutTree.get()!)).not.toContain('logs')
    expect($logsOpen.get()).toBe(false)
  })
})
