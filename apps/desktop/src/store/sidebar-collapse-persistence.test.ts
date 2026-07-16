import { computed } from 'nanostores'
import { beforeEach, describe, expect, it, vi } from 'vitest'

// Behavior contract for the PRODUCTION binding path: the titlebar sidebar
// button owns the sessions PANE (bindPaneVisibility), not a physical side.
// Drive the REAL stores, then re-import them (fresh module state reading
// persisted localStorage) to simulate a ⌃R reload. `bind` applies the exact
// wiring the controller ships: $sidebarOpen ∨ narrow drives the hidden set,
// close/open route back through setSidebarOpen so the toggle stays truthful.
async function loadStores() {
  const layout = await import('./layout')
  const tree = await import('@/components/pane-shell/tree/store')

  return {
    layout,
    tree,
    bind: () =>
      tree.bindPaneVisibility(
        'sessions',
        computed([layout.$sidebarOpen, tree.$narrowViewport], (open, narrow) => open || narrow),
        () => layout.setSidebarOpen(false),
        () => layout.setSidebarOpen(true)
      ),
    sessionsHidden: () => tree.$hiddenTreePanes.get().has('sessions')
  }
}

const reload = () => vi.resetModules() // fresh modules; localStorage is the carry-over

describe('sidebar pane visibility persistence', () => {
  beforeEach(() => {
    window.localStorage.clear()
    vi.resetModules()
  })

  it('restores a hidden sidebar after a reload', async () => {
    const s1 = await loadStores()
    s1.bind()
    s1.layout.setSidebarOpen(false)
    expect(s1.sessionsHidden()).toBe(true)

    reload()
    const s2 = await loadStores()
    expect(s2.layout.$sidebarOpen.get()).toBe(false) // persisted open:false survives
    s2.bind()
    expect(s2.sessionsHidden()).toBe(true) // and re-hides the pane
  })

  // A sidebar HIDDEN before a reset must be reopened by the reset ("restore
  // everything") THROUGH ITS OWNING STORE, so the button state agrees;
  // otherwise the stale-hidden state flips the next ⌘B into a SHOW, and the
  // user's hide never appears to persist.
  it('reset reopens a hidden sidebar through its store, so a later hide persists across reload', async () => {
    const s1 = await loadStores()
    const { group, split } = await import('@/components/pane-shell/tree/model')
    s1.tree.declareDefaultTree(split('row', [group(['sessions']), group(['workspace'])], [1, 3]))
    s1.bind()

    s1.layout.setSidebarOpen(false) // hidden BEFORE the reset
    expect(s1.sessionsHidden()).toBe(true)

    s1.tree.resetLayoutTree() // mod-click reset — restores everything, sidebar shown again
    expect(s1.layout.$sidebarOpen.get()).toBe(true) // …through the store, not around it
    expect(s1.sessionsHidden()).toBe(false)

    s1.layout.toggleSidebarOpen() // ⌘B now genuinely hides
    expect(s1.layout.$sidebarOpen.get()).toBe(false)

    reload()
    const s2 = await loadStores()
    expect(s2.layout.$sidebarOpen.get()).toBe(false)
    s2.bind()
    expect(s2.sessionsHidden()).toBe(true)
  })

  // Narrow viewports hand visibility to the edge overlays, and NarrowOverlays
  // skips hidden panes entirely — a closed pane must leave the hidden set
  // while narrow (else its titlebar button and ⌘B reveal nothing), and return
  // to the store's remembered state when the viewport widens again.
  it('keeps a closed sidebar revealable on a narrow viewport', async () => {
    const s1 = await loadStores()
    s1.bind()

    s1.layout.setSidebarOpen(false)
    expect(s1.sessionsHidden()).toBe(true)

    s1.tree.$narrowViewport.set(true) // shrink below the collapse breakpoint
    expect(s1.sessionsHidden()).toBe(false) // overlay candidate again

    s1.tree.$narrowViewport.set(false) // widen: remembered hide re-applies
    expect(s1.sessionsHidden()).toBe(true)
    expect(s1.layout.$sidebarOpen.get()).toBe(false) // intent never clobbered
  })
})
