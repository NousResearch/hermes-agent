// Loaded once at collection, outside every test's budget: vi.resetModules() keeps each module's
// transformed code, so the fresh copy a test imports only re-evaluates it. Without this the first
// test pays the cold transform of the whole graph and times out on a loaded runner.
import '@/components/pane-shell/tree/store'
import '@/store/layout'

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// Pane-owned visibility follows the flip, independently of positional
// keyboard/titlebar toggles. Closing files in the "Sidebar right" layout
// must not hide sessions too. Regression for that layout rendering only chat.

describe('side toggles follow the flip', () => {
  beforeEach(() => {
    window.localStorage.clear()
    vi.resetModules()
  })

  afterEach(() => {
    vi.resetModules()
  })

  it('collapses the edge the sidebar / file tree actually sit on', async () => {
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
