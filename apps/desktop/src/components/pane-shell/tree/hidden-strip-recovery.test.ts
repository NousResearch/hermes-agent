import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// Ground-truth repro for "hidden tab strip becomes an inescapable dead end for
// chat tabs" (upstream issue: crash-stale or user-set `headerHidden: true` on
// the main zone). The tab strip is the ONLY affordance that shows/switches/
// closes chat tabs, and the only "show header" control lives in the header's
// own context menu — so once the strip is hidden:
//
//   1. revealTreePane('session-tile:X') fronted the tab INVISIBLY — every
//      sidebar click looked dead (nothing on screen changed), and
//   2. adoption re-pinned `headerHidden: true` onto the zone (the "standing
//      preference" branch), so even NEW tabs arrived invisible.
//
// Fix under test: revealing or adopting a SESSION pane (workspace /
// session-tile:*) into a header-hidden zone force-shows the bar; tool panels
// (terminal, logs, …) keep honoring the zone's standing preference.

describe('hidden tab strip recovery for chat tabs', () => {
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
      area: 'panes',
      data: { placement: 'main', uncloseable: true },
      id: 'workspace',
      render: () => null,
      title: 'chat'
    })

    tree.declareDefaultTree(
      model.split(
        'row',
        [model.group(['workspace'], { active: 'workspace', id: 'grp-main' })],
        [1]
      )
    )
    tree.watchContributedPanes()

    return { model, registry, tree }
  }

  const mainGroup = (tree: Awaited<ReturnType<typeof setup>>['tree']) => {
    const root = tree.$layoutTree.get()!
    const find = (n: import('@/components/pane-shell/tree/model').LayoutNode): import('@/components/pane-shell/tree/model').GroupNode | null => {
      if (n.type === 'group') {
        return n.panes.includes('workspace') || n.panes.some(p => p.startsWith('session-tile:')) ? n : null
      }

      for (const child of n.children) {
        const hit = find(child)

        if (hit) {
          return hit
        }
      }

      return null
    }

    return find(root)!
  }

  it('revealTreePane un-hides the strip for a session tile', async () => {
    const { registry, tree } = await setup()

    registry.register({
      area: 'panes',
      data: { dock: { pane: 'workspace', pos: 'center' }, placement: 'main' },
      id: 'session-tile:s1',
      render: () => null,
      title: 's1'
    })
    expect(mainGroup(tree).panes).toContain('session-tile:s1')

    // The trap state: strip explicitly hidden while a chat tab exists.
    tree.setTreeGroupHeaderHidden(mainGroup(tree).id, true)
    expect(mainGroup(tree).headerHidden).toBe(true)

    // A sidebar click on the open session funnels through revealTreePane.
    // Pre-fix: the tab fronted invisibly (headerHidden stayed true) — the
    // user saw a dead click with no way to ever show the strip again.
    tree.revealTreePane('session-tile:s1')

    expect(mainGroup(tree).headerHidden).toBe(false)
    expect(mainGroup(tree).active).toBe('session-tile:s1')
  })

  it('adopting a NEW session tile into a hidden-strip zone shows the strip', async () => {
    const { registry, tree } = await setup()

    tree.setTreeGroupHeaderHidden(mainGroup(tree).id, true)
    expect(mainGroup(tree).headerHidden).toBe(true)

    // "Open in new tab" registers a new tile pane; adoption docks it into the
    // main zone. Pre-fix the adoption path re-pinned the zone's hidden flag
    // (standing-preference branch), so the new tab arrived invisible.
    registry.register({
      area: 'panes',
      data: { dock: { pane: 'workspace', pos: 'center' }, placement: 'main' },
      id: 'session-tile:s2',
      render: () => null,
      title: 's2'
    })

    expect(mainGroup(tree).panes).toContain('session-tile:s2')
    expect(mainGroup(tree).headerHidden).toBe(false)
  })

  it('tool panels still honor the standing hidden preference on adoption', async () => {
    const { registry, tree } = await setup()

    tree.setTreeGroupHeaderHidden(mainGroup(tree).id, true)

    // A tool pane (NOT a session strip pane) adopted into the same zone must
    // keep the user's hidden bar — that's the legitimate standing preference
    // the original branch protects (close/reopen of terminal, logs, …).
    registry.register({
      area: 'panes',
      data: { dock: { pane: 'workspace', pos: 'center' }, placement: 'main' },
      id: 'scratchpad',
      render: () => null,
      title: 'scratchpad'
    })

    expect(mainGroup(tree).panes).toContain('scratchpad')
    expect(mainGroup(tree).headerHidden).toBe(true)
  })
})
