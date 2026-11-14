import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { LayoutNode } from '@/components/pane-shell/tree/model'
import { PREVIEW_TILE_PREFIX } from '@/store/preview-tile-id'

// The drift test below imports the preview tile watcher to exercise REAL
// id minting. Its render/console leaves are never invoked there (the mirror
// only registers pane contributions), and their component graph is heavy —
// mock them so this file stays light under parallel full-suite runs.
vi.mock('@/app/chat/right-rail/preview', () => ({ PreviewTilePane: () => null }))
vi.mock('@/app/chat/right-rail/preview-console-store', () => ({ forgetPreviewConsole: () => {} }))

// Preview tiles are transient: their panes must never reach the persisted
// layout tree. A tree saved WITH them bakes splits/weights around tabs the
// next session may not restore (artifact previews never persist; the Browser
// tab re-keys to a singleton), so a hand-built pane arrangement re-assembles
// differently on every boot. `persist()` must scrub them from the STORED copy
// while the LIVE tree keeps them for the session, and a stored tree written by
// an older build must heal on load.

const TREE_KEY = 'hermes.desktop.layoutTree.v2'

function storedTree(): LayoutNode | null {
  const raw = window.localStorage.getItem(TREE_KEY)

  return raw ? (JSON.parse(raw) as LayoutNode) : null
}

describe('ephemeral preview panes are scrubbed from the persisted tree', () => {
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

    registry.register({
      id: 'workspace',
      area: 'panes',
      title: 'chat',
      data: { placement: 'main', uncloseable: true },
      render: () => null
    })

    const registerPreview = (id: string, pos: 'center' | 'right' = 'right') =>
      registry.register({
        id,
        area: 'panes',
        title: 'Preview',
        data: { placement: 'main', dock: { pane: 'workspace', pos } },
        render: () => null
      })

    const registerFiles = () =>
      registry.register({
        id: 'files',
        area: 'panes',
        title: 'Files',
        data: { placement: 'side', dock: { pane: 'workspace', pos: 'right' } },
        render: () => null
      })

    tree.declareDefaultTree(model.group(['workspace'], { id: 'grp-main' }))
    tree.watchContributedPanes()

    return { model, registerFiles, registerPreview, registry, tree }
  }

  it('persistTree() scrubs preview panes from storage but keeps them live', async () => {
    const { model, registerPreview, tree } = await setup()

    registerPreview('preview-tile:file:notes')

    // Live tree: the pane is docked beside workspace (a real split).
    const live = tree.$layoutTree.get()!

    expect(live.type).toBe('split')
    expect(model.allPaneIds(live)).toContain('preview-tile:file:notes')

    tree.persistTree()

    // Stored copy: the split collapses back to the bare workspace group.
    const stored = storedTree()!

    expect(stored.type).toBe('group')
    expect(model.allPaneIds(stored)).toEqual(['workspace'])

    // The live tree is untouched — the pane stays for THIS session.
    expect(model.allPaneIds(tree.$layoutTree.get()!)).toContain('preview-tile:file:notes')
  })

  it('scrubbing a stacked preview tab does not strand its zone active', async () => {
    const { model, tree } = await setup()

    // The preview tab IS the active tab of its zone — the scrubbed stored copy
    // must fall back to a real pane instead of persisting a dangling active id.
    tree.$layoutTree.set(
      model.group(['workspace', 'preview-tile:artifact:abc'], { id: 'grp-main', active: 'preview-tile:artifact:abc' })
    )

    tree.persistTree()

    const stored = storedTree()!

    expect(model.allPaneIds(stored)).toEqual(['workspace'])

    if (stored.type !== 'group') {
      throw new Error(`expected a group, got ${stored.type}`)
    }

    expect(stored.active).toBe('workspace')

    // Live tree untouched.
    expect(model.allPaneIds(tree.$layoutTree.get()!)).toEqual(['workspace', 'preview-tile:artifact:abc'])
  })

  it('non-ephemeral panes and their weights persist unchanged', async () => {
    const { model, registerFiles, tree } = await setup()

    registerFiles()

    const root = tree.$layoutTree.get()!

    tree.setTreeSplitWeights(root.id, [3, 1])
    tree.persistTree()

    const stored = storedTree()!

    expect(model.allPaneIds(stored)).toEqual(['workspace', 'files'])

    if (stored.type !== 'split') {
      throw new Error(`expected a split, got ${stored.type}`)
    }

    expect(stored.weights).toEqual([3, 1])
  })

  it('persisting a preview-free tree never materializes the pane-id list', async () => {
    const { model, registerFiles, tree } = await setup()

    // Adoption of `files` itself runs `allPaneIds` (dock-anchor lookup) — that
    // is expected. The assertion covers the EVERY-COMMIT persist that follows:
    // with no preview tile in the tree it must early-out before building ids.
    registerFiles()

    const spy = vi.spyOn(model, 'allPaneIds')

    tree.persistTree()

    expect(spy).not.toHaveBeenCalled()

    spy.mockRestore()
  })

  it('the ephemeral predicate reads the namespace the minting side owns', async () => {
    await setup()

    const { isEphemeralPane } = await import('@/components/pane-shell/tree/store-persist')

    // One definition: the scrubber keys off the same prefix the preview tile
    // mints from (`store/preview-tile-id`), not a second copy of the literal.
    expect(isEphemeralPane(`${PREVIEW_TILE_PREFIX}:file:notes`)).toBe(true)
    expect(isEphemeralPane('workspace')).toBe(false)
    expect(isEphemeralPane('route-tile:/skills')).toBe(false)
    // The bare namespace is not itself a pane id.
    expect(isEphemeralPane(PREVIEW_TILE_PREFIX)).toBe(false)
  })

  // `vi.resetModules()` forces the preview-tile graph to be re-imported inside
  // this test, and that graph is heavy — under a full parallel suite the cold
  // import alone blows the 15s default. Same reason `layout-flipped-sides` and
  // `connections` raise theirs.
  it('pane ids minted by the preview tile mirror satisfy the shared predicate', { timeout: 30_000 }, async () => {
    const { registry } = await setup()
    const { isEphemeralPane } = await import('@/components/pane-shell/tree/store-persist')

    const { $previewTabs } = await import('@/store/preview')
    const { watchPreviewTiles } = await import('@/app/chat/preview-tile')

    watchPreviewTiles()

    $previewTabs.set([
      {
        id: 'file:notes',
        target: { kind: 'file', label: 'notes', source: 'C:/notes.md', url: 'file:///C:/notes.md' }
      }
    ])

    const ephemeralPaneIds = registry
      .getArea('panes')
      .map(contribution => contribution.id)
      .filter(id => isEphemeralPane(id))

    expect(ephemeralPaneIds).toEqual(['preview-tile:file:notes'])
  })

  it('a stored tree written by an older build heals on load', async () => {
    // Seed storage the way a pre-fix build would have persisted it: the live
    // split including a stale artifact preview pane that can never come back.
    const { model } = await setup()

    window.localStorage.setItem(
      TREE_KEY,
      JSON.stringify(
        model.split('row', [
          model.group(['workspace'], { id: 'grp-main' }),
          model.group(['preview-tile:artifact:stale'], { id: 'grp-stale' })
        ])
      )
    )

    // Fresh module state: `$layoutTree` initializes from the seeded storage.
    vi.resetModules()
    const tree = await import('@/components/pane-shell/tree/store')

    const booted = tree.$layoutTree.get()!

    // The ghost preview pane is gone before anything renders…
    expect(model.allPaneIds(booted)).toEqual(['workspace'])
    expect(booted.type).toBe('group')

    // …and the live session can still open a preview: adoption re-docks it,
    // and the NEXT persist keeps storage clean.
    const { registry } = await import('@/contrib/registry')

    registry.register({
      id: 'preview-tile:url:browser',
      area: 'panes',
      title: 'Browser',
      data: { placement: 'main', dock: { pane: 'workspace', pos: 'right' } },
      render: () => null
    })
    tree.declareDefaultTree(booted)
    tree.watchContributedPanes()

    expect(model.allPaneIds(tree.$layoutTree.get()!)).toContain('preview-tile:url:browser')

    tree.persistTree()
    expect(model.allPaneIds(storedTree()!)).not.toContain('preview-tile:url:browser')
  })

  it('switching interface mode stores the scrubbed tree, not the live one', async () => {
    const { model, registerPreview, tree } = await setup()

    registerPreview('preview-tile:file:notes')
    expect(model.allPaneIds(tree.$layoutTree.get()!)).toContain('preview-tile:file:notes')

    // The mode switch captures every scoped atom through its codec: a second
    // writer next to persist(), and it must apply the same policy.
    const { setInterfaceMode } = await import('@/store/interface-mode')

    setInterfaceMode('simple')

    expect(model.allPaneIds(storedTree()!)).toEqual(['workspace'])
  })

  it('a stored tree heals on load whichever module loads first', async () => {
    const { model } = await setup()

    window.localStorage.setItem(
      TREE_KEY,
      JSON.stringify(
        model.split('row', [
          model.group(['workspace'], { id: 'grp-main' }),
          model.group(['preview-tile:artifact:ghost'], { id: 'grp-ghost' })
        ])
      )
    )

    // The preview store graph loads ahead of the tree store: the load-time
    // scrub must not depend on which side of the graph evaluates first.
    vi.resetModules()
    await import('@/store/preview-explicit')
    const tree = await import('@/components/pane-shell/tree/store')

    expect(model.allPaneIds(tree.$layoutTree.get()!)).toEqual(['workspace'])
  })
})
