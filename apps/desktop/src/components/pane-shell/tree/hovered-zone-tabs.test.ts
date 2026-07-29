import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// The tab verbs (⌘1…⌘9, ⌃Tab, ⌘W / ⌘T) target the zone under the POINTER when
// there is one, so hovering a pane and hitting ⌘2 lands there without clicking
// in first. Pointer off the panes falls back to the last interacted zone.

describe('hovered zone retargets the tab verbs', () => {
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

    for (const id of ['workspace', 'session-tile:a', 'session-tile:b', 'session-tile:c']) {
      registry.register({
        area: 'panes',
        data: id === 'workspace' ? { placement: 'main', uncloseable: true } : { placement: 'main' },
        id,
        render: () => null,
        title: id
      })
    }

    // Two chat zones side by side, each a real strip (≥2 shown tabs).
    tree.declareDefaultTree(
      model.split('row', [
        model.group(['workspace', 'session-tile:a'], { active: 'workspace', id: 'grp-main' }),
        model.group(['session-tile:b', 'session-tile:c'], { active: 'session-tile:b', id: 'grp-side' })
      ])
    )

    const activeOf = (id: string) => model.findGroup(tree.$layoutTree.get()!, id)?.active ?? null

    return { activeOf, model, tree }
  }

  it('⌘2 switches the HOVERED zone even while the other one holds focus', async () => {
    const { activeOf, tree } = await setup()

    tree.noteActiveTreeGroup('grp-main')
    tree.noteHoveredTreeGroup('grp-side')

    expect(tree.activateTreeTabSlot(2)).toBe(true)
    expect(activeOf('grp-side')).toBe('session-tile:c')
    // The focused zone is untouched — the pointer won the target, not both.
    expect(activeOf('grp-main')).toBe('workspace')

    // Same key, pointer moved: the other zone's slot 2.
    tree.noteHoveredTreeGroup('grp-main')
    expect(tree.activateTreeTabSlot(2)).toBe(true)
    expect(activeOf('grp-main')).toBe('session-tile:a')
  })

  it('pointer off every zone falls back to the focused one', async () => {
    const { activeOf, tree } = await setup()

    tree.noteActiveTreeGroup('grp-side')
    tree.noteHoveredTreeGroup('grp-main')
    tree.noteHoveredTreeGroup(null)

    expect(tree.activateTreeTabSlot(2)).toBe(true)
    expect(activeOf('grp-side')).toBe('session-tile:c')
    expect(activeOf('grp-main')).toBe('workspace')
  })

  it('⌃Tab and the ⌘T / ⌘W family follow the same hovered zone', async () => {
    const { activeOf, model, tree } = await setup()

    tree.noteActiveTreeGroup('grp-main')
    tree.noteHoveredTreeGroup('grp-side')

    expect(tree.cycleTreeTabInFocusedZone(1)).toBe(true)
    expect(activeOf('grp-side')).toBe('session-tile:c')
    expect(activeOf('grp-main')).toBe('workspace')

    expect(tree.focusedSessionTabAnchor()).toBe('session-tile:c')

    expect(tree.closeFocusedSessionTab()).toBe(true)
    expect(model.allPaneIds(tree.$layoutTree.get()!)).not.toContain('session-tile:c')
    expect(model.allPaneIds(tree.$layoutTree.get()!)).toContain('workspace')
  })
})
