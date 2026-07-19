import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// Ground-truth repro for "reactive unhides silently re-open a user-collapsed
// side": `setTreePaneHidden` is the primitive that backs `bindPaneVisibility`
// (e.g. `bindPaneVisibility('files', $hasWorkspace)`). The original
// implementation auto-called `revealTreePane` on every unhide, which expanded
// the parent column even when the user had explicitly collapsed it — so a
// workspace transition (`$currentCwd` flipping on session create / resume)
// forced the right sidebar open and persisted that open state to localStorage.
//
// We mirror controller.tsx exactly: register the `files` pane with
// `placement: 'right'`, declare the default tree, bind the right side to
// the file-browser open store, then collapse the right sidebar and trigger
// the reactive unhide via the same primitive `bindPaneVisibility` invokes.

describe('reactive pane unhide', () => {
  beforeEach(() => {
    window.localStorage.clear()
    vi.resetModules()
  })

  afterEach(() => {
    vi.resetModules()
  })

  async function setupWithFiles() {
    const tree = await import('@/components/pane-shell/tree/store')
    const layout = await import('@/store/layout')
    const model = await import('@/components/pane-shell/tree/model')
    const { registry } = await import('@/contrib/registry')

    // Register `files` like controller.tsx does — placement 'right' is what
    // makes `treeSideOfPane('files')` return 'right' (and therefore what
    // makes the buggy `revealTreePane` auto-expand the right column).
    registry.register({
      id: 'files',
      area: 'panes',
      title: 'files',
      data: { placement: 'right' },
      render: () => null
    })

    // Declare a minimal default tree mirroring the production DEFAULT_TREE's
    // row shape (sessions | workspace | right-column-with-files).
    tree.declareDefaultTree(
      model.split('row', [
        model.group(['sessions'], { id: 'grp-sessions' }),
        model.group(['workspace'], { id: 'grp-main' }),
        model.split('column', [
          model.split('row', [
            model.group(['files'], { id: 'grp-files' })
          ], [1], 'spl-rail'),
          model.group(['terminal'], { id: 'grp-terminal' })
        ], [1.6, 1], 'spl-right')
      ], [0.85, 3, 1.6])
    )

    // Mirror controller.tsx:512.
    tree.bindTreeSideVisibility('right', layout.$fileBrowserOpen, layout.setFileBrowserOpen)

    return { tree, layout }
  }

  it('reactive unhide does NOT expand a user-collapsed side', async () => {
    const { tree, layout } = await setupWithFiles()

    // Simulate the production scenario: a detached chat has no cwd, so
    // `bindPaneVisibility('files', $hasWorkspace)` calls `setTreePaneHidden(
    // 'files', true)` (hidden). Then the user collapses the right sidebar.
    // Then a new session acquires a cwd and the workspace wiring unhides
    // `files`. That last step is what the bug corrupts.
    tree.setTreePaneHidden('files', true)
    expect(tree.$hiddenTreePanes.get().has('files')).toBe(true)

    layout.setFileBrowserOpen(false)
    expect(layout.$fileBrowserOpen.get()).toBe(false)
    expect(tree.$collapsedTreeSides.get().has('right')).toBe(true)

    // Workspace flips: `bindPaneVisibility('files', $hasWorkspace)` calls
    // `setTreePaneHidden('files', false)` — model it directly.
    tree.setTreePaneHidden('files', false)

    // The pane unhides…
    expect(tree.$hiddenTreePanes.get().has('files')).toBe(false)

    // …but the user-collapsed side MUST stay collapsed. Before the fix, the
    // unhide called `revealTreePane`, which saw the side was collapsed and
    // called `setFileBrowserOpen(true)` — silently re-opening the right
    // sidebar on every session create / resume.
    expect(layout.$fileBrowserOpen.get()).toBe(false)
    expect(tree.$collapsedTreeSides.get().has('right')).toBe(true)
  })

  it('reactive HIDE (hidden=true) leaves the side flag alone', async () => {
    const { tree, layout } = await setupWithFiles()

    layout.setFileBrowserOpen(false)
    expect(tree.$collapsedTreeSides.get().has('right')).toBe(true)

    // A reactive HIDE (workspace goes away) must not flip the side either —
    // collapsing the side is the user's domain; hiding individual panes
    // inside it is the workspace domain.
    tree.setTreePaneHidden('files', true)

    expect(tree.$hiddenTreePanes.get().has('files')).toBe(true)
    expect(layout.$fileBrowserOpen.get()).toBe(false)
    expect(tree.$collapsedTreeSides.get().has('right')).toBe(true)
  })

  it('explicit setFileBrowserOpen still expands the side (user toggle)', async () => {
    const { tree, layout } = await setupWithFiles()

    layout.setFileBrowserOpen(false)
    expect(tree.$collapsedTreeSides.get().has('right')).toBe(true)

    // The user toggles the sidebar back open — this MUST still work; the
    // binding mirrors `$fileBrowserOpen` to `$collapsedTreeSides`.
    layout.setFileBrowserOpen(true)
    expect(layout.$fileBrowserOpen.get()).toBe(true)
    expect(tree.$collapsedTreeSides.get().has('right')).toBe(false)
  })

  it('reactive unhide does not invoke the right side opener directly', async () => {
    const { tree, layout } = await setupWithFiles()

    // Spy on the opener that `revealTreePane` would call when expanding a
    // collapsed side — the bug is exactly this call firing on reactive unhide.
    const openerSpy = vi.fn()
    tree.bindTreeSideVisibility('right', layout.$fileBrowserOpen, openerSpy)

    layout.setFileBrowserOpen(false)
    expect(tree.$collapsedTreeSides.get().has('right')).toBe(true)

    tree.setTreePaneHidden('files', true)
    expect(tree.$hiddenTreePanes.get().has('files')).toBe(true)

    openerSpy.mockClear()

    // Reactive unhide fires via the same primitive the workspace wiring uses.
    tree.setTreePaneHidden('files', false)

    // The opener MUST NOT be called — before the fix, the auto-reveal would
    // call `setFileBrowserOpen(true)` via this opener.
    expect(tree.$hiddenTreePanes.get().has('files')).toBe(false)
    expect(openerSpy).not.toHaveBeenCalled()
  })
})
