import { beforeEach, expect, it, vi } from 'vitest'

import type { LayoutNode } from './model'

beforeEach(() => {
  window.localStorage.clear()
  vi.resetModules()
})

async function boot() {
  const mode = await import('@/store/interface-mode')
  const panes = await import('@/store/panes')
  const layout = await import('@/store/layout')
  const tree = await import('./store')
  const model = await import('./model')
  const { registry } = await import('@/contrib/registry')
  const presets = await import('./presets')
  const { registerLayoutPresets, DEFAULT_TREE, BASIC_TREE } = await import('@/app/contrib/layout-presets')
  const terminal = await import('@/app/right-sidebar/store')
  const { bindLayoutSides } = await import('@/app/contrib/layout-sides')

  for (const [id, placement] of [
    ['sessions', 'left'],
    ['workspace', 'main'],
    ['files', 'right'],
    ['review', 'right'],
    ['terminal', 'bottom']
  ] as const) {
    registry.register({ id, area: 'panes', data: { placement }, render: () => null })
  }
  registry.register({
    id: 'bots',
    area: 'panes',
    render: () => null,
    data: { placement: 'left', dock: { pane: 'sessions', pos: 'center' } }
  })
  registerLayoutPresets()
  tree.declareDefaultTree(DEFAULT_TREE, BASIC_TREE)
  tree.watchContributedPanes()
  bindLayoutSides()
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
    () => terminal.setTerminalTakeover(true),
    mode.$showsAdvancedChrome
  )

  const apply = (id: string) => {
    const preset = registry.getArea('layouts').find(p => p.id === id)!
    presets.applyLayoutPreset(id, preset.data as LayoutNode)
  }
  const snapshot = () => ({
    tree: structuredClone(tree.$layoutTree.get()),
    panes: structuredClone(panes.$paneStates.get()),
    preset: tree.$activePresetId.get(),
    hidden: [...tree.$hiddenStripTabs.get()],
    dismissed: [...tree.$dismissedPanes.get()],
    placed: [...tree.$userPlacedPanes.get()],
    collapsed: [...tree.$collapsedTreeSides.get()].sort(),
    flipped: layout.$panesFlipped.get()
  })

  return { mode, panes, layout, tree, model, registry, presets, terminal, apply, snapshot }
}

it('restores independently customized modes through real pane bindings and reloads', async () => {
  let app = await boot()
  const { mode, panes, layout, tree, model, registry, presets, terminal } = app
  const decks = registry.getArea('layouts').filter(p => presets.layoutPresetTier(p.id) === 'advanced')

  // Every Advanced preset must survive both Simple sidebar arrangements.
  for (const deck of decks) {
    for (const simpleId of ['sidebar-left', 'sidebar-right']) {
      mode.setInterfaceMode('advanced')
      app.apply(deck.id)
      tree.setStripTabHidden('bots', true)
      panes.setPaneWidthOverride('chat-sidebar', 317)
      panes.setPaneHeightOverride('terminal', 243)
      tree.setTreeGroupMinimized(model.findGroupOfPane(tree.$layoutTree.get()!, 'terminal')!.id, true)
      layout.setSidebarOpen(false)
      const advanced = app.snapshot()
      const saved = window.localStorage.getItem('hermes.desktop.layoutTree.v2')

      mode.setInterfaceMode('simple')
      app.apply(simpleId)
      panes.setPaneWidthOverride('chat-sidebar', 225)
      tree.setStripTabHidden('bots', false)
      layout.setSidebarOpen(true)
      const simple = app.snapshot()
      expect(window.localStorage.getItem('hermes.desktop.layoutTree.v2')).toBe(saved)

      terminal.$terminalInjection.set('pending fixture command')
      layout.$rightRailActiveTabId.set('file:fixture.txt')
      mode.setInterfaceMode('advanced')
      expect(app.snapshot()).toEqual(advanced)
      expect(tree.$hiddenTreePanes.get().has('bots')).toBe(true)
      expect(terminal.$terminalInjection.get()).toBe('pending fixture command')
      expect(layout.$rightRailActiveTabId.get()).toBe('file:fixture.txt')
      mode.setInterfaceMode('simple')
      expect(app.snapshot()).toEqual(simple)
      expect(tree.$hiddenTreePanes.get().has('bots')).toBe(false)
      mode.setInterfaceMode('simple')
      expect(app.snapshot()).toEqual(simple)
    }
  }

  // A moved/customized tree, dismissals and an unfinished resize survive too.
  mode.setInterfaceMode('advanced')
  app.apply('default')
  tree.moveTreePane('files', { groupId: model.findGroupOfPane(tree.$layoutTree.get()!, 'workspace')!.id, pos: 'center' })
  tree.dismissTreePane('review')
  const root = tree.$layoutTree.get()!

  if (root.type !== 'split') {
    throw new Error('fixture must have a split')
  }

  tree.setTreeSplitWeights(root.id, root.weights.map((weight, i) => weight + i))
  const advanced = app.snapshot()
  mode.setInterfaceMode('simple')
  const simple = app.snapshot()
  vi.resetModules()
  app = await boot()
  expect(app.snapshot()).toEqual({ ...simple, tree: app.model.normalize(simple.tree!) })
  app.mode.setInterfaceMode('advanced')
  expect(app.snapshot()).toEqual({ ...advanced, tree: app.model.normalize(advanced.tree!) })

  // Newly opened shared work is adopted, not lost when restoring an older tree.
  app.mode.setInterfaceMode('simple')
  app.registry.register({ id: 'new-shared-pane', area: 'panes', data: { placement: 'main' }, render: () => null })
  app.mode.setInterfaceMode('advanced')
  expect(app.model.allPaneIds(app.tree.$layoutTree.get()!)).toContain('new-shared-pane')
})

it.each(['advanced', 'simple'] as const)('keeps legacy %s data and never re-inherits it after migration', async initialMode => {
  const { group, split, normalize } = await import('./model')
  const legacyTree = split(
    'row',
    [group(['sessions', 'bots']), group(['workspace']), group(['files', 'review', 'terminal'])],
    [2, 8, 3]
  )
  const legacyPanes = { 'chat-sidebar': { open: false, widthOverride: 301 }, 'file-browser': { open: true } }
  const raw = JSON.stringify(legacyTree)
  window.localStorage.setItem('hermes.desktop.layoutTree.v2', raw)
  window.localStorage.setItem('hermes.desktop.paneStates.v1', JSON.stringify(legacyPanes))
  window.localStorage.setItem('hermes.desktop.layoutPreset.active', 'custom')
  window.localStorage.setItem('hermes.desktop.hiddenStripTabs.v1', '["bots"]')
  if (initialMode === 'simple') {
    window.localStorage.setItem('hermes.desktop.interfaceMode.v1', 'simple')
  }

  const { mode, tree, panes, layout } = await boot()
  expect(mode.$interfaceMode.get()).toBe(initialMode)
  expect(panes.$paneStates.get()).toEqual(legacyPanes)
  expect(tree.$hiddenStripTabs.get().has('bots')).toBe(true)
  // Boot collapse may update the tree, so preserve the settled arrangement.
  const settled = structuredClone(tree.$layoutTree.get())
  mode.setInterfaceMode(initialMode === 'simple' ? 'advanced' : 'simple')
  expect(tree.$collapsedTreeSides.get().has(layout.sidebarSide())).toBe(!layout.$sidebarOpen.get())
  expect(tree.$collapsedTreeSides.get().has(layout.fileBrowserSide())).toBe(!layout.$fileBrowserOpen.get())
  mode.setInterfaceMode(initialMode)
  expect(tree.$layoutTree.get()).toEqual(settled)
  expect(panes.$paneStates.get()).toEqual(legacyPanes)

  mode.setInterfaceMode('simple')
  tree.setStripTabHidden('bots', false)
  panes.setPaneWidthOverride('chat-sidebar', 210)
  vi.resetModules()
  const reloaded = await boot()
  expect(reloaded.tree.$hiddenStripTabs.get().size).toBe(0)
  expect(reloaded.panes.$paneStates.get()['chat-sidebar'].widthOverride).toBe(210)
  reloaded.mode.setInterfaceMode('advanced')
  expect(reloaded.panes.$paneStates.get()).toEqual(legacyPanes)
  expect(reloaded.model.allPaneIds(reloaded.tree.$layoutTree.get()!)).toEqual(reloaded.model.allPaneIds(normalize(legacyTree)))
})
