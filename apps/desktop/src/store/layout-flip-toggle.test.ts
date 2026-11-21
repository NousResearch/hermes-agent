import { beforeEach, expect, it, vi } from 'vitest'

import { allPaneIds, group, type LayoutNode, split } from '@/components/pane-shell/tree/model'

// #131701 — swap-sides worked once per window-state change and then died.
// The $panesFlipped listener mirrors the tree, but the $layoutTree subscription
// re-derives the flag from sessionsOnRight() and overwrites it. When the
// mirrored tree does not move `sessions` across `workspace` (column-heavy
// trees), the derived write reverts the flag and the next press writes the
// same value — a silent no-op — while for trees without a sessions pane the
// gesture mirrored nothing at all. The gesture must stay authoritative and
// every press must apply.

beforeEach(() => {
  window.localStorage.clear()
  vi.resetModules()
})

async function boot() {
  const mode = await import('@/store/interface-mode')
  const layout = await import('@/store/layout')
  const tree = await import('@/components/pane-shell/tree/store')
  const { registry } = await import('@/contrib/registry')
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

  return { layout, tree }
}

it('every flip press mirrors, even when the mirror keeps sessions on its side', { timeout: 120_000 }, async () => {
  const { layout, tree } = await boot()

  // Column-of-rows: the root column keeps its order, so mirroring leaves
  // sessionsOnRight() unchanged — exactly the tree whose derived write used
  // to eat the gesture.
  const columnHeavy = (): LayoutNode =>
    split('column', [
      split('row', [group(['sessions']), group(['files'])]),
      split('row', [group(['workspace']), group(['review'])])
    ])

  tree.$layoutTree.set(columnHeavy())
  expect(layout.$panesFlipped.get()).toBe(false)

  const beforePress1 = JSON.stringify(tree.$layoutTree.get())
  layout.togglePanesFlipped()
  expect(layout.$panesFlipped.get()).toBe(true)
  const afterPress1 = JSON.stringify(tree.$layoutTree.get())
  expect(afterPress1).not.toBe(beforePress1)

  layout.togglePanesFlipped()
  expect(layout.$panesFlipped.get()).toBe(false)
  const afterPress2 = JSON.stringify(tree.$layoutTree.get())
  expect(afterPress2).not.toBe(afterPress1)
  // An involution: two presses return the arrangement.
  expect(JSON.parse(afterPress2)).toEqual(JSON.parse(beforePress1))

  // Layout edits still REMAP the flag instead of mirroring the tree back.
  const dragged = split('row', [group(['workspace']), group(['sessions'])])
  tree.$layoutTree.set(dragged)
  expect(layout.$panesFlipped.get()).toBe(true)
  // The tree was not mirrored in response to the remap.
  expect(allPaneIds(tree.$layoutTree.get()!)).toEqual(['workspace', 'sessions'])
})

it('a tree without a sessions pane still mirrors on every press', { timeout: 120_000 }, async () => {
  const { layout, tree } = await boot()

  // sessionsOnRight() is null here (no sessions pane); the derivation cannot
  // veto the gesture, so the press must apply.
  tree.$layoutTree.set(split('row', [group(['files']), group(['workspace']), group(['review'])]))
  expect(layout.$panesFlipped.get()).toBe(false)

  const beforePress1 = JSON.stringify(tree.$layoutTree.get())
  layout.togglePanesFlipped()
  expect(layout.$panesFlipped.get()).toBe(true)
  expect(JSON.stringify(tree.$layoutTree.get())).not.toBe(beforePress1)

  const afterPress1 = JSON.stringify(tree.$layoutTree.get())
  layout.togglePanesFlipped()
  expect(layout.$panesFlipped.get()).toBe(false)
  const afterPress2 = JSON.stringify(tree.$layoutTree.get())
  expect(afterPress2).not.toBe(afterPress1)
  expect(JSON.parse(afterPress2)).toEqual(JSON.parse(beforePress1))
})
