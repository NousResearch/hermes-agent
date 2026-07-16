import { beforeEach, describe, expect, it, vi } from 'vitest'

/**
 * Pane-scoped toggle ownership — the product contract for the titlebar
 * controls: the left sidebar button owns ONLY `sessions`, the right sidebar
 * button owns ONLY `files`, and each plugin pane toggle (Browser / QC via
 * `host.panes.setOpen/toggle`) owns only its own pane. Toggling one pane must
 * never change any other pane's hidden state, persisted open flag, or tree
 * membership.
 *
 * Drives the REAL stores, the REAL `bindPaneVisibility` export, and the REAL
 * plugin SDK; the sessions/files bindings mirror the controller's production
 * wiring (same computeds, incl. the narrow-viewport opt-out). Fresh module
 * state per test via vi.resetModules(); localStorage is the only carry-over
 * across a "reload".
 */

const BROWSER = 'visual-workbench:browser'
const QC = 'visual-workbench:qc'
const PLUGIN_SOURCE = 'plugin:visual-workbench'
const ALL_PANES = ['sessions', 'workspace', 'files', BROWSER, QC]

async function boot() {
  const tree = await import('@/components/pane-shell/tree/store')
  const model = await import('@/components/pane-shell/tree/model')
  const panes = await import('@/store/panes')
  const layout = await import('@/store/layout')
  const { registry } = await import('@/contrib/registry')
  const sdk = await import('@/sdk')
  const { atom, computed } = await import('nanostores')

  // Default layout: sessions | main | files | browser | qc — files, browser
  // and qc all share the right physical side, which is exactly the shape the
  // old side-collapse binding used to break.
  tree.declareDefaultTree(
    model.split(
      'row',
      [
        model.group(['sessions']),
        model.group(['workspace']),
        model.group(['files']),
        model.group([BROWSER]),
        model.group([QC])
      ],
      [1, 3, 1, 1, 1]
    )
  )

  registry.registerMany([
    { id: 'sessions', area: 'panes', title: 'Sessions', data: { placement: 'left' }, render: () => null },
    {
      id: 'workspace',
      area: 'panes',
      title: 'Chat',
      data: { placement: 'main', uncloseable: true },
      render: () => null
    },
    { id: 'files', area: 'panes', title: 'Files', data: { placement: 'right' }, render: () => null },
    {
      id: BROWSER,
      area: 'panes',
      source: PLUGIN_SOURCE,
      title: 'Browser',
      data: { placement: 'right' },
      render: () => null
    },
    {
      id: QC,
      area: 'panes',
      source: PLUGIN_SOURCE,
      title: 'Quality Control',
      data: { placement: 'right' },
      render: () => null
    }
  ])

  tree.watchContributedPanes()

  // The controller's production wiring, driven through the real export:
  // sessions/files opt OUT of the hidden set on narrow viewports so the edge
  // overlays own their visibility below the breakpoint.
  const $hasWorkspace = atom(true)
  tree.bindPaneVisibility(
    'sessions',
    computed([layout.$sidebarOpen, tree.$narrowViewport], (open, narrow) => open || narrow),
    () => layout.setSidebarOpen(false),
    () => layout.setSidebarOpen(true)
  )
  tree.bindPaneVisibility(
    'files',
    computed(
      [layout.$fileBrowserOpen, $hasWorkspace, tree.$narrowViewport],
      (open, workspace, narrow) => (open || narrow) && workspace
    ),
    () => layout.setFileBrowserOpen(false),
    () => layout.setFileBrowserOpen(true)
  )

  const hidden = () => tree.$hiddenTreePanes.get()
  const openOf = (id: string) => panes.$paneOpen(id).get()

  const treePaneIds = () => {
    const current = tree.$layoutTree.get()

    return current ? [...model.allPaneIds(current)].sort() : []
  }

  /** Visibility snapshot of every pane EXCEPT the ones being exercised. */
  const othersSnapshot = (except: string[]) =>
    ALL_PANES.filter(id => !except.includes(id)).map(id => ({ id, hidden: hidden().has(id) }))

  return { $hasWorkspace, hidden, layout, openOf, othersSnapshot, panes, sdk, tree, treePaneIds }
}

describe('pane-scoped toggle independence', () => {
  beforeEach(() => {
    window.localStorage.clear()
    vi.resetModules()
  })

  it('right sidebar toggle changes files only; every other pane keeps its visibility', async () => {
    const s = await boot()
    s.layout.setFileBrowserOpen(true)

    const before = s.othersSnapshot(['files'])
    const treeBefore = s.treePaneIds()

    s.layout.setFileBrowserOpen(false) // right sidebar OFF
    expect(s.hidden().has('files')).toBe(true)
    expect(s.othersSnapshot(['files'])).toEqual(before)
    expect(s.treePaneIds()).toEqual(treeBefore) // hides, never removes

    s.layout.setFileBrowserOpen(true) // right sidebar ON
    expect(s.hidden().has('files')).toBe(false)
    expect(s.othersSnapshot(['files'])).toEqual(before)
    expect(s.treePaneIds()).toEqual(treeBefore)
  })

  it('left sidebar toggle changes sessions only', async () => {
    const s = await boot()

    const before = s.othersSnapshot(['sessions'])

    s.layout.setSidebarOpen(false)
    expect(s.hidden().has('sessions')).toBe(true)
    expect(s.layout.$sidebarOpen.get()).toBe(false)
    expect(s.othersSnapshot(['sessions'])).toEqual(before)

    s.layout.setSidebarOpen(true)
    expect(s.hidden().has('sessions')).toBe(false)
    expect(s.othersSnapshot(['sessions'])).toEqual(before)
  })

  it('Browser and QC toggles stay independent through every pairwise combination', async () => {
    const s = await boot()

    const combos: Array<[boolean, boolean]> = [
      [true, true],
      [true, false],
      [false, true],
      [false, false],
      [true, true]
    ]

    const chromeBefore = s.othersSnapshot([BROWSER, QC])
    const treeBefore = s.treePaneIds()

    for (const [browserOpen, qcOpen] of combos) {
      s.sdk.host.panes.setOpen(BROWSER, browserOpen)
      s.sdk.host.panes.setOpen(QC, qcOpen)

      // Each plugin pane's rendered-tree state tracks exactly its own flag.
      expect(s.hidden().has(BROWSER)).toBe(!browserOpen)
      expect(s.hidden().has(QC)).toBe(!qcOpen)

      // The SDK atom, persisted pane store, and hidden tree agree per pane.
      expect(s.sdk.host.panes.open(BROWSER).get()).toBe(browserOpen)
      expect(s.sdk.host.panes.open(QC).get()).toBe(qcOpen)
      expect(s.openOf(BROWSER)).toBe(browserOpen)
      expect(s.openOf(QC)).toBe(qcOpen)

      // Core chrome (sessions/workspace/files) never moves with plugin panes.
      expect(s.othersSnapshot([BROWSER, QC])).toEqual(chromeBefore)
      expect(s.treePaneIds()).toEqual(treeBefore)
    }
  })

  it('host.panes.toggle flips only its own pane', async () => {
    const s = await boot()

    const before = s.othersSnapshot([BROWSER])

    s.sdk.host.panes.toggle(BROWSER)
    expect(s.hidden().has(BROWSER)).toBe(true)
    expect(s.openOf(BROWSER)).toBe(false)
    expect(s.othersSnapshot([BROWSER])).toEqual(before)

    s.sdk.host.panes.toggle(BROWSER)
    expect(s.hidden().has(BROWSER)).toBe(false)
    expect(s.openOf(BROWSER)).toBe(true)
    expect(s.othersSnapshot([BROWSER])).toEqual(before)
  })

  it('refuses SDK writes to core chrome panes — their stores stay authoritative', async () => {
    const s = await boot()

    s.sdk.host.panes.setOpen('sessions', false)
    expect(s.hidden().has('sessions')).toBe(false) // untouched
    expect(s.layout.$sidebarOpen.get()).toBe(true)

    s.sdk.host.panes.setOpen('files', true)
    expect(s.layout.$fileBrowserOpen.get()).toBe(false) // untouched
  })

  it('losing the workspace hides files without clobbering the remembered intent or touching Browser/QC', async () => {
    const s = await boot()
    s.layout.setFileBrowserOpen(true)
    expect(s.hidden().has('files')).toBe(false)

    const before = s.othersSnapshot(['files'])

    s.$hasWorkspace.set(false) // project detached
    expect(s.hidden().has('files')).toBe(true)
    expect(s.layout.$fileBrowserOpen.get()).toBe(true) // intent survives
    expect(s.othersSnapshot(['files'])).toEqual(before)

    s.$hasWorkspace.set(true) // project back — remembered intent restores files
    expect(s.hidden().has('files')).toBe(false)
    expect(s.othersSnapshot(['files'])).toEqual(before)
  })

  it('narrow viewport opts sessions/files out of the hidden set without touching their stores or plugin panes', async () => {
    const s = await boot()
    s.layout.setSidebarOpen(false)
    s.sdk.host.panes.setOpen(BROWSER, false)
    expect(s.hidden().has('sessions')).toBe(true)

    s.tree.$narrowViewport.set(true) // breakpoint crossed
    expect(s.hidden().has('sessions')).toBe(false) // overlay owns it now
    expect(s.hidden().has('files')).toBe(false)
    expect(s.layout.$sidebarOpen.get()).toBe(false) // remembered intent intact
    expect(s.hidden().has(BROWSER)).toBe(true) // plugin panes unaffected
    expect(s.openOf(BROWSER)).toBe(false)

    s.tree.$narrowViewport.set(false) // wide again — remembered state re-drives
    expect(s.hidden().has('sessions')).toBe(true)
    expect(s.layout.$sidebarOpen.get()).toBe(false)
  })

  it('a hidden left sidebar persists across a reload and still hides only sessions', async () => {
    const s1 = await boot()
    s1.sdk.host.panes.setOpen(BROWSER, true)
    s1.layout.setSidebarOpen(false)
    expect(s1.hidden().has('sessions')).toBe(true)

    vi.resetModules() // "restart": fresh modules, persisted localStorage carries over

    const s2 = await boot()
    expect(s2.layout.$sidebarOpen.get()).toBe(false) // persisted intent survives
    expect(s2.hidden().has('sessions')).toBe(true)
    expect(s2.hidden().has('files')).toBe(true) // file browser defaults closed
    expect(s2.hidden().has(BROWSER)).toBe(false) // plugin pane untouched by the restart
    expect(s2.hidden().has(QC)).toBe(false)
    expect(s2.hidden().has('workspace')).toBe(false)
  })
})
