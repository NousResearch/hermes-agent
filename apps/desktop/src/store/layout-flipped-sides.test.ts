import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// An edge belongs to the pane that sits on it. Mirrored (⌘\ or the Simple
// "Sidebar right" layout), the sessions sidebar is on the RIGHT — ⌘B must
// still mean the sidebar and ⌘J the file tree, or a resting file tree folds
// the sidebar away with it. Regression for the "Sidebar right" layout
// rendering only the chat.

describe('side toggles follow the flip', () => {
  beforeEach(() => {
    window.localStorage.clear()
    vi.resetModules()
  })

  afterEach(() => {
    vi.resetModules()
  })

  it('collapses the edge the sidebar / file tree actually sit on', { timeout: 30_000 }, async () => {
    const tree = await import('@/components/pane-shell/tree/store')
    const layout = await import('@/store/layout')

    expect(layout.sidebarSide()).toBe('left')
    expect(layout.fileBrowserSide()).toBe('right')

    layout.$panesFlipped.set(true)

    expect(layout.sidebarSide()).toBe('right')
    expect(layout.fileBrowserSide()).toBe('left')

    layout.setFileBrowserOpen(false)
    expect(tree.$collapsedTreeSides.get().has('left')).toBe(true)
    expect(tree.$collapsedTreeSides.get().has('right')).toBe(false)

    layout.setSidebarOpen(false)
    expect(tree.$collapsedTreeSides.get().has('right')).toBe(true)

    layout.setFileBrowserOpen(true)
    layout.setSidebarOpen(true)
    expect(tree.$collapsedTreeSides.get().size).toBe(0)
  })
})
