import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// ⌘1…⌘9 / ⌃Tab must index the same tabs the strip paints. A chrome-hidden
// pane (Focus layout's `files`) stays in `group.panes` but isn't a chip —
// indexing the raw array made ⌘2 land on the strip's first session tab.

describe('activateTreeTabSlot indexes shown panes only', () => {
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

    for (const id of ['workspace', 'files', 'session-tile:a', 'session-tile:b']) {
      registry.register({
        area: 'panes',
        data: id === 'files' ? { collapsible: true, placement: 'side' } : { placement: 'main' },
        id,
        render: () => null,
        title: id
      })
    }

    // Focus-ish packing: files rides with workspace; two session tiles stacking
    // of top of that — pane order in the tree: workspace, files, A, B.
    tree.declareDefaultTree(
      model.group(['workspace', 'files', 'session-tile:a', 'session-tile:b'], {
        active: 'workspace',
        id: 'grp-main'
      })
    )
    tree.noteActiveTreeGroup('grp-main')
    tree.setTreePaneHidden('files', true)

    const activeOf = () => model.findGroup(tree.$layoutTree.get()!, 'grp-main')?.active ?? null

    return { activeOf, tree }
  }

  it('⌘1 is workspace and ⌘2 is the first SESSION tab when files is hidden', async () => {
    const { activeOf, tree } = await setup()

    expect(tree.activateTreeTabSlot(1)).toBe('workspace')
    expect(activeOf()).toBe('workspace')

    expect(tree.activateTreeTabSlot(2)).toBe('session-tile:a')
    expect(activeOf()).toBe('session-tile:a')

    expect(tree.activateTreeTabSlot(3)).toBe('session-tile:b')
    expect(activeOf()).toBe('session-tile:b')
  })

  it('does not claim a slot past the visible count (falls through to profile switch)', async () => {
    const { tree } = await setup()

    // Shown: workspace + A + B → 3. Slot 4 would have been `B` on the raw array
    // (workspace, files, A, B) before this fix — now it correctly refuses.
    expect(tree.activateTreeTabSlot(4)).toBeNull()
  })

  it('⌃Tab cycles only visible chips', async () => {
    const { activeOf, tree } = await setup()

    expect(tree.cycleTreeTabInFocusedZone(1)).toBe('session-tile:a')
    expect(activeOf()).toBe('session-tile:a')

    expect(tree.cycleTreeTabInFocusedZone(1)).toBe('session-tile:b')
    expect(activeOf()).toBe('session-tile:b')

    expect(tree.cycleTreeTabInFocusedZone(1)).toBe('workspace')
    expect(activeOf()).toBe('workspace')
  })
})
