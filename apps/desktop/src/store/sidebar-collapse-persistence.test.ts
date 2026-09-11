import { beforeEach, describe, expect, it, vi } from 'vitest'

// Ground-truth repro for "hiding the sidebar doesn't persist on reload": drive
// the REAL stores, then re-import them (fresh module state reading persisted
// localStorage) to simulate a ⌃R reload. `bind` mirrors the controller wiring.
async function loadStores() {
  const layout = await import('./layout')
  const tree = await import('@/components/pane-shell/tree/store')

  return {
    layout,
    tree,
    bind: () => tree.bindTreeSideVisibility('left', layout.$sidebarOpen, layout.setSidebarOpen),
    leftCollapsed: () => tree.$collapsedTreeSides.get().has('left')
  }
}

const reload = () => vi.resetModules() // fresh modules; localStorage is the carry-over

describe('sidebar collapse persistence', () => {
  beforeEach(() => {
    window.localStorage.clear()
    vi.resetModules()
  })

  it('restores a hidden sidebar after a reload', async () => {
    const s1 = await loadStores()
    s1.bind()
    s1.layout.setSidebarOpen(false)
    expect(s1.leftCollapsed()).toBe(true)

    reload()
    const s2 = await loadStores()
    expect(s2.layout.$sidebarOpen.get()).toBe(false) // persisted open:false survives
    s2.bind()
    expect(s2.leftCollapsed()).toBe(true) // and re-collapses
  })

  // The reported repro: a sidebar HIDDEN before a reset must be reopened by the
  // reset ("restore everything"); otherwise the stale-hidden state flips the
  // next ⌘B into a SHOW, and the user's hide never persists.
  it('reset reopens a hidden sidebar, so a later hide persists across reload', async () => {
    const s1 = await loadStores()
    const { group, split } = await import('@/components/pane-shell/tree/model')
    s1.tree.declareDefaultTree(split('row', [group(['sessions']), group(['workspace'])], [1, 3]))
    s1.bind()

    s1.layout.setSidebarOpen(false) // hidden BEFORE the reset
    expect(s1.leftCollapsed()).toBe(true)

    s1.tree.resetLayoutTree() // ⌘⇧ reset — restores everything, sidebar shown again
    expect(s1.layout.$sidebarOpen.get()).toBe(true)
    expect(s1.leftCollapsed()).toBe(false)

    s1.layout.toggleSidebarOpen() // ⌘B now genuinely hides
    expect(s1.layout.$sidebarOpen.get()).toBe(false)

    reload()
    const s2 = await loadStores()
    expect(s2.layout.$sidebarOpen.get()).toBe(false)
    s2.bind()
    expect(s2.leftCollapsed()).toBe(true)
  })

  it('explicit open restores hidden strip tabs and minimized groups only on that physical side', async () => {
    for (const flipped of [false, true]) {
      window.localStorage.clear()
      reload()
      const { layout, tree, bind } = await loadStores()
      const { findGroup, group, split } = await import('@/components/pane-shell/tree/model')
      const { registry } = await import('@/contrib/registry')

      const disposers = [
        registry.register({ area: 'panes', id: 'sessions', data: { placement: 'left' } }),
        registry.register({ area: 'panes', id: 'bots', data: { placement: 'left' } }),
        registry.register({ area: 'panes', id: 'workspace', data: { placement: 'main' } }),
        registry.register({ area: 'panes', id: 'files', data: { placement: 'right' } }),
        registry.register({ area: 'panes', id: 'review', data: { placement: 'right' } })
      ]

      try {
        const sidebar = group(['sessions', 'bots'], { active: 'bots', id: 'sidebar' })
        const main = group(['workspace'])
        const other = group(['files', 'review'], { id: 'other' })
        tree.declareDefaultTree(split('row', flipped ? [other, main, sidebar] : [sidebar, main, other]))
        bind()
        tree.bindTreeSideVisibility('right', layout.$fileBrowserOpen, layout.setFileBrowserOpen)
        layout.setFileBrowserOpen(true)
        tree.setStripTabHidden('sessions', true)
        tree.setStripTabHidden('review', true)
        tree.setTreeGroupMinimized('sidebar', true)
        tree.setTreeGroupMinimized('other', true)
        const open = flipped ? layout.setFileBrowserOpen : layout.setSidebarOpen

        open(true) // already true: must not depend on a nanostores notification
        expect(tree.isStripTabHidden('sessions')).toBe(false)
        expect(tree.$hiddenTreePanes.get().has('sessions')).toBe(false)
        expect(tree.isStripTabHidden('review')).toBe(true)
        expect(findGroup(tree.$layoutTree.get()!, 'sidebar')).toMatchObject({ active: 'bots', minimized: false })
        expect(findGroup(tree.$layoutTree.get()!, 'other')?.minimized).toBe(true)
        expect(JSON.parse(window.localStorage.getItem('hermes.desktop.hiddenStripTabs.v1')!)).toEqual(['review'])
        const restored = tree.$layoutTree.get()
        open(true)
        expect(tree.$layoutTree.get()).toBe(restored)
        tree.setStripTabHidden('sessions', true)
        const toggle = flipped ? layout.toggleFileBrowserOpen : layout.toggleSidebarOpen
        toggle()
        toggle()
        expect(tree.isStripTabHidden('sessions')).toBe(true) // ordinary toggles preserve the tab choice
      } finally {
        disposers.forEach(dispose => dispose())
      }
    }
  })

  it('recovers the minimized physical side after reload without changing its active tab', async () => {
    for (const flipped of [false, true]) {
      window.localStorage.clear()
      reload()
      const s1 = await loadStores()
      const { group, split } = await import('@/components/pane-shell/tree/model')
      const sidebar = group(['sessions', 'bots'], { active: 'bots', id: 'sidebar' })
      const main = group(['workspace'])
      const files = group(['files'], { id: 'files-zone' })
      s1.tree.declareDefaultTree(split('row', flipped ? [files, main, sidebar] : [sidebar, main, files]))
      s1.tree.setTreeGroupMinimized(sidebar.id, true)
      s1.layout.setFileBrowserOpen(true)

      reload()
      const { layout, tree, bind } = await loadStores()
      const { findGroup } = await import('@/components/pane-shell/tree/model')
      const { registry } = await import('@/contrib/registry')

      const disposers = [
        registry.register({ area: 'panes', id: 'sessions', data: { placement: 'left' } }),
        registry.register({ area: 'panes', id: 'bots', data: { placement: 'left' } }),
        registry.register({ area: 'panes', id: 'workspace', data: { placement: 'main' } }),
        registry.register({ area: 'panes', id: 'files', data: { placement: 'right' } })
      ]

      try {
        bind()
        tree.bindTreeSideVisibility('right', layout.$fileBrowserOpen, layout.setFileBrowserOpen)
        const side = flipped ? 'right' : 'left'
        const toggle = flipped ? layout.toggleFileBrowserOpen : layout.toggleSidebarOpen
        const $open = flipped ? layout.$fileBrowserOpen : layout.$sidebarOpen
        const otherToggle = flipped ? layout.toggleSidebarOpen : layout.toggleFileBrowserOpen

        // Boot respects minimize; the OTHER physical button must not restore it.
        expect(findGroup(tree.$layoutTree.get()!, sidebar.id)?.minimized).toBe(true)
        otherToggle()
        expect(tree.$collapsedTreeSides.get().has(flipped ? 'left' : 'right')).toBe(true)
        expect(findGroup(tree.$layoutTree.get()!, sidebar.id)?.minimized).toBe(true)

        toggle()
        expect($open.get()).toBe(true)
        expect(tree.$collapsedTreeSides.get().has(side)).toBe(false)
        expect(findGroup(tree.$layoutTree.get()!, sidebar.id)).toMatchObject({ active: 'bots', minimized: false })
        toggle()
        expect($open.get()).toBe(false)
        expect(tree.$collapsedTreeSides.get().has(side)).toBe(true)
        toggle()
        expect($open.get()).toBe(true)
        expect(tree.$collapsedTreeSides.get().has(side)).toBe(false)
        expect(findGroup(tree.$layoutTree.get()!, sidebar.id)?.active).toBe('bots')
      } finally {
        disposers.forEach(dispose => dispose())
      }
    }
  })
})
