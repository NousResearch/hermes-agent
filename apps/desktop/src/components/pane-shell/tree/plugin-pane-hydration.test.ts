import { beforeEach, describe, expect, it, vi } from 'vitest'

/**
 * Plugin pane hydration + restart persistence — the adoption bridge contract:
 * a runtime plugin pane that ARRIVES visible must report open through
 * `host.panes.open(id)` (fresh profile), and a pane the user closed must come
 * back HIDDEN after a full restart (persisted `open: false` wins over the
 * adoption default). Atom, persisted snapshot, and hidden-tree membership must
 * agree at every step.
 *
 * Real stores + real SDK; vi.resetModules() is the restart (fresh module
 * state, localStorage is the carry-over).
 */

const BROWSER = 'visual-workbench:browser'
const QC = 'visual-workbench:qc'
const PLUGIN_SOURCE = 'plugin:visual-workbench'
const PANE_STATES_KEY = 'hermes.desktop.paneStates.v1'

async function bootCore() {
  const tree = await import('@/components/pane-shell/tree/store')
  const model = await import('@/components/pane-shell/tree/model')
  const panes = await import('@/store/panes')
  const { registry } = await import('@/contrib/registry')

  tree.declareDefaultTree(model.split('row', [model.group(['sessions']), model.group(['workspace'])], [1, 3]))

  registry.registerMany([
    { id: 'sessions', area: 'panes', title: 'Sessions', data: { placement: 'left' }, render: () => null },
    {
      id: 'workspace',
      area: 'panes',
      title: 'Chat',
      data: { placement: 'main', uncloseable: true },
      render: () => null
    }
  ])

  tree.watchContributedPanes()

  // The plugin registers AFTER core layout boot — the runtime-loaded shape.
  const registerPluginPanes = () => {
    registry.registerMany([
      {
        id: BROWSER,
        area: 'panes',
        source: PLUGIN_SOURCE,
        title: 'Browser',
        data: { dock: { pane: 'workspace', pos: 'right' }, placement: 'right' },
        render: () => null
      },
      {
        id: QC,
        area: 'panes',
        source: PLUGIN_SOURCE,
        title: 'Quality Control',
        data: { dock: { pane: BROWSER, pos: 'right' }, placement: 'right' },
        render: () => null
      }
    ])
  }

  const inTree = (id: string) => {
    const current = tree.$layoutTree.get()

    return Boolean(current && model.allPaneIds(current).includes(id))
  }

  const persistedOpen = (id: string): boolean | undefined =>
    (JSON.parse(window.localStorage.getItem(PANE_STATES_KEY) ?? '{}') as Record<string, { open?: boolean }>)[id]?.open

  return { inTree, panes, persistedOpen, registerPluginPanes, tree }
}

describe('plugin pane hydration and restart persistence', () => {
  beforeEach(() => {
    window.localStorage.clear()
    vi.resetModules()
  })

  it('fresh profile: an adopted plugin pane renders visible AND reports open', async () => {
    const s = await bootCore()
    s.registerPluginPanes()

    for (const id of [BROWSER, QC]) {
      expect(s.inTree(id)).toBe(true) // adopted into the rendered tree
      expect(s.tree.$hiddenTreePanes.get().has(id)).toBe(false) // rendered visible
      expect(s.panes.$paneOpen(id).get()).toBe(true) // ...and says so
      expect(s.persistedOpen(id)).toBe(true) // seeded exactly once, persisted
    }
  })

  it('persisted open:false hydrates as hidden — never a visible pane that reports closed', async () => {
    window.localStorage.setItem(PANE_STATES_KEY, JSON.stringify({ [BROWSER]: { open: false } }))
    vi.resetModules()

    const s = await bootCore()
    s.registerPluginPanes()

    expect(s.inTree(BROWSER)).toBe(true)
    expect(s.tree.$hiddenTreePanes.get().has(BROWSER)).toBe(true) // stays hidden
    expect(s.panes.$paneOpen(BROWSER).get()).toBe(false)

    // The sibling pane with no persisted preference still defaults visible.
    expect(s.tree.$hiddenTreePanes.get().has(QC)).toBe(false)
    expect(s.panes.$paneOpen(QC).get()).toBe(true)
  })

  it('closing through the SDK persists, and the pane stays hidden after a restart', async () => {
    const s1 = await bootCore()
    s1.registerPluginPanes()
    const sdk1 = await import('@/sdk')

    sdk1.host.panes.setOpen(BROWSER, false)
    expect(s1.tree.$hiddenTreePanes.get().has(BROWSER)).toBe(true)
    expect(s1.persistedOpen(BROWSER)).toBe(false)

    vi.resetModules() // full restart: fresh modules, persisted state carries

    const s2 = await bootCore()
    s2.registerPluginPanes()
    const sdk2 = await import('@/sdk')

    expect(s2.tree.$hiddenTreePanes.get().has(BROWSER)).toBe(true) // persisted-false stays hidden
    expect(sdk2.host.panes.open(BROWSER).get()).toBe(false)
    expect(s2.tree.$hiddenTreePanes.get().has(QC)).toBe(false) // sibling untouched
    expect(sdk2.host.panes.open(QC).get()).toBe(true)

    // Re-opening after the restart round-trips the same three-way agreement.
    sdk2.host.panes.setOpen(BROWSER, true)
    expect(s2.tree.$hiddenTreePanes.get().has(BROWSER)).toBe(false)
    expect(sdk2.host.panes.open(BROWSER).get()).toBe(true)
    expect(s2.persistedOpen(BROWSER)).toBe(true)
  })
})

describe('layout reset vs pane preferences', () => {
  beforeEach(() => {
    window.localStorage.clear()
    vi.resetModules()
  })

  it('reset reopens store-bound chrome panes but keeps a closed plugin pane closed', async () => {
    const s = await bootCore()
    s.registerPluginPanes()
    const layout = await import('@/store/layout')
    const { computed } = await import('nanostores')
    const sdk = await import('@/sdk')

    // Production wiring: sessions is a store-bound chrome pane with an opener.
    s.tree.bindPaneVisibility(
      'sessions',
      computed([layout.$sidebarOpen, s.tree.$narrowViewport], (open, narrow) => open || narrow),
      () => layout.setSidebarOpen(false),
      () => layout.setSidebarOpen(true)
    )

    layout.setSidebarOpen(false)
    sdk.host.panes.setOpen(BROWSER, false)
    expect(s.tree.$hiddenTreePanes.get().has('sessions')).toBe(true)
    expect(s.tree.$hiddenTreePanes.get().has(BROWSER)).toBe(true)

    s.tree.resetLayoutTree() // "restore everything"

    // Chrome pane: reopened THROUGH its store, so the toggle stays truthful.
    expect(layout.$sidebarOpen.get()).toBe(true)
    expect(s.tree.$hiddenTreePanes.get().has('sessions')).toBe(false)

    // Plugin pane: no opener — its own persisted preference survives the reset.
    expect(s.tree.$hiddenTreePanes.get().has(BROWSER)).toBe(true)
    expect(sdk.host.panes.open(BROWSER).get()).toBe(false)
    expect(s.persistedOpen(BROWSER)).toBe(false)
    expect(sdk.host.panes.open(QC).get()).toBe(true) // sibling untouched
  })
})
