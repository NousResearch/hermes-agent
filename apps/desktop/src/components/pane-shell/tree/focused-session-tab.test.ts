import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// ⌘W / ⌘T / ⌘⇧T / the strip "+" used to hardcode the workspace's zone while
// ⌘1…⌘9 followed the interacted zone — so every tab verb missed a SECOND chat
// zone the user was working in. They now resolve the same focused zone.

describe('focused chat zone drives the tab verbs', () => {
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
        data: id === 'workspace' ? { placement: 'main', uncloseable: true } : { placement: 'main' },
        id,
        render: () => null,
        title: id
      })
    }

    // Two chat zones side by side: main holds the workspace, the second holds
    // its own session tabs (a split-off stack).
    tree.declareDefaultTree(
      model.split('row', [
        model.group(['workspace'], { active: 'workspace', id: 'grp-main' }),
        model.group(['session-tile:a', 'session-tile:b'], { active: 'session-tile:b', id: 'grp-side' })
      ])
    )

    return { model, tree }
  }

  it('⌘T anchors the new tab to the focused zone, not always the workspace', async () => {
    const { tree } = await setup()

    tree.noteActiveTreeGroup('grp-side')
    expect(tree.focusedSessionTabAnchor()).toBe('session-tile:b')

    tree.noteActiveTreeGroup('grp-main')
    expect(tree.focusedSessionTabAnchor()).toBe('workspace')
  })

  it('a non-chat zone (files/terminal focus) falls back to the workspace', async () => {
    const { model, tree } = await setup()

    tree.declareDefaultTree(
      model.split('row', [
        model.group(['workspace'], { active: 'workspace', id: 'grp-main' }),
        model.group(['files'], { active: 'files', id: 'grp-files' })
      ])
    )
    tree.noteActiveTreeGroup('grp-files')

    expect(tree.focusedSessionTabAnchor()).toBe('workspace')
    // ⌘W must not close the file tree just because it holds focus.
    expect(tree.closeFocusedSessionTab()).toBe(false)
  })

  it('⌘W closes the focused zone active tab and leaves the workspace alone', async () => {
    const { model, tree } = await setup()

    tree.noteActiveTreeGroup('grp-side')
    expect(tree.closeFocusedSessionTab()).toBe(true)
    expect(model.allPaneIds(tree.$layoutTree.get()!)).not.toContain('session-tile:b')

    // Back on main: the uncloseable workspace is a no-op (the caller promotes
    // a stacked session instead of closing the window).
    tree.noteActiveTreeGroup('grp-main')
    expect(tree.closeFocusedSessionTab()).toBe(false)
    expect(model.allPaneIds(tree.$layoutTree.get()!)).toContain('workspace')
  })
})
