import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// Focus layout packs `workspace` and `files` into one tab group. A reactive
// unhide of `files` (cwd arrives on the first reply / new session) must NOT
// steal the active tab away from the workspace/new-session the user is looking
// at — the "files panel hijacks the tab ~1s after the first reply" bug.
// frontPaneInGroup only takes the active slot when the current active pane is
// not itself showable, so it still picks a valid tab when nothing is visible.

describe('reactive unhide in a shared (Focus) group', () => {
  beforeEach(() => {
    window.localStorage.clear()
    vi.resetModules()
  })

  afterEach(() => {
    vi.resetModules()
  })

  async function setupFocusGroup() {
    const tree = await import('@/components/pane-shell/tree/store')
    const model = await import('@/components/pane-shell/tree/model')
    const { registry } = await import('@/contrib/registry')

    for (const id of ['workspace', 'files']) {
      registry.register({ id, area: 'panes', title: id, data: {}, render: () => null })
    }

    // Focus layout: sessions | (workspace + files share one group), workspace
    // fronted — mirrors FOCUS_TREE's single-group packing in controller.tsx.
    tree.declareDefaultTree(
      model.split(
        'row',
        [
          model.group(['sessions'], { id: 'grp-sessions' }),
          model.group(['workspace', 'files'], { active: 'workspace', id: 'grp-focus' })
        ],
        [1, 4.6]
      )
    )

    const activeOf = (paneId: string) => model.findGroupOfPane(tree.$layoutTree.get()!, paneId)?.active ?? null

    return { activeOf, tree }
  }

  it('does not steal the active tab from workspace when files unhides', async () => {
    const { activeOf, tree } = await setupFocusGroup()

    // Detached chat: no cwd → files hidden, workspace is the active tab.
    tree.setTreePaneHidden('files', true)
    expect(activeOf('workspace')).toBe('workspace')

    // First reply adopts a cwd → the workspace wiring unhides files.
    tree.setTreePaneHidden('files', false)

    expect(tree.$hiddenTreePanes.get().has('files')).toBe(false)
    // The user stays on the new session; files must not hijack the tab.
    expect(activeOf('workspace')).toBe('workspace')
  })

  it('still fronts the unhidden pane when the active tab is not showable', async () => {
    const { activeOf, tree } = await setupFocusGroup()

    // Nothing valid showing: the active pane is itself hidden.
    tree.setTreePaneHidden('workspace', true)
    tree.setTreePaneHidden('files', true)

    tree.setTreePaneHidden('files', false)

    // With no showable active pane, the reactive unhide picks files so the
    // group fronts a valid tab (preserves the #65375 "visible next time" intent).
    expect(activeOf('files')).toBe('files')
  })
})
