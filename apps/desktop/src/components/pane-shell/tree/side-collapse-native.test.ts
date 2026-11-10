import { afterEach, expect, it, vi } from 'vitest'

/** The real tree and side stores in a native Desktop renderer: the Electron
 *  preload's bridge is present and no browser-host marker is set. */
async function bootNative() {
  vi.resetModules()
  ;(window as unknown as { hermesDesktop?: unknown }).hermesDesktop = {}
  const tree = await import('./store')
  const layout = await import('@/store/layout')
  const terminal = await import('@/app/right-sidebar/store')
  const { registry } = await import('@/contrib/registry')
  const { DEFAULT_TREE } = await import('@/app/contrib/layout-presets')
  const { isBrowserHostedDesktop } = await import('@/lib/platform')

  for (const [id, placement] of [
    ['sessions', 'left'],
    ['workspace', 'main'],
    ['files', 'right'],
    ['review', 'right'],
    ['terminal', 'bottom']
  ]) {
    registry.register({ id, area: 'panes', title: id, data: { placement }, render: () => null })
  }

  tree.declareDefaultTree(DEFAULT_TREE)
  tree.bindTreeSideVisibility('left', layout.$sidebarOpen, layout.setSidebarOpen)
  tree.bindTreeSideVisibility('right', layout.$fileBrowserOpen, layout.setFileBrowserOpen)
  tree.bindPaneVisibility(
    'files',
    layout.$fileBrowserOpen,
    () => layout.setFileBrowserOpen(false),
    () => layout.setFileBrowserOpen(true)
  )
  tree.bindToolPaneCollapse(
    'terminal',
    terminal.$terminalTakeover,
    () => terminal.setTerminalTakeover(false),
    () => terminal.setTerminalTakeover(true)
  )

  return { isBrowserHostedDesktop, layout, terminal, tree }
}

afterEach(() => {
  delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
  window.localStorage.clear()
})

// A pane is on screen only when the side holding it is open. In native Desktop
// a terminal whose right side was collapsed (Files closed with ⌘J) reads as
// hidden to the statusbar toggle and the terminal keybinds, though its own
// store still says open, and its toggle reveals it on the first press instead
// of closing a terminal the user cannot see.
it('reports a pane on a collapsed side as hidden in native Desktop and reveals it on the first toggle', async () => {
  const { isBrowserHostedDesktop, layout, terminal, tree } = await bootNative()

  expect(isBrowserHostedDesktop()).toBe(false)
  tree.togglePaneVisible('terminal')
  expect(tree.isPaneVisible('terminal')).toBe(true)

  const $visible = tree.$paneVisible('terminal')
  layout.setFileBrowserOpen(false)

  expect(tree.$collapsedTreeSides.get().has('right')).toBe(true)
  expect(terminal.$terminalTakeover.get()).toBe(true)
  expect(tree.isPaneVisible('terminal')).toBe(false)
  expect($visible.get()).toBe(false)

  tree.togglePaneVisible('terminal')

  expect(tree.isPaneVisible('terminal')).toBe(true)
  expect($visible.get()).toBe(true)
  expect(terminal.$terminalTakeover.get()).toBe(true)
}, 30_000)
