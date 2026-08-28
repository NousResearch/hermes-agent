import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// Enforced dock invariants: a pane whose dock hint carries `enforce: true`
// (Bot Mode's Bots pane, Cronjobs) is guaranteed present and sanely docked
// at boot. A stale stacked arrangement WITHOUT a user-placed record still
// re-homes — persisted layouts otherwise pin that broken shape forever,
// because adoption only ever places panes MISSING from the tree. A pane the
// user explicitly dragged (`$userPlacedPanes`) whose persisted group still
// resolves keeps that spot across boots. An unusable persisted spot (the
// pane's group no longer resolves) re-homes even when the drag record
// exists. Unlike the retired one-time heal, a burned heal token does not
// exempt the pane. The invariant is boot-scoped, so an intra-session drag
// sticks until the next launch.

const TREE_KEY = 'hermes.desktop.layoutTree.v2'
const USER_PLACED_KEY = 'hermes.desktop.userPlacedPanes.v1'
const LEGACY_HEAL_KEY = 'hermes.desktop.paneDockHeals.v1'

// The shipped regression shape: sessions and bots as SIBLING groups in a
// column (the old `pos: 'bottom'` split), workspace beside them.
const stackedTree = {
  type: 'split',
  id: 'root',
  orientation: 'row',
  weights: [1, 3],
  children: [
    {
      type: 'split',
      id: 'left-col',
      orientation: 'column',
      weights: [1, 1],
      children: [
        { type: 'group', id: 'g-sessions', panes: ['sessions'], active: 'sessions' },
        { type: 'group', id: 'g-bots', panes: ['hermes-bots:pane'], active: 'hermes-bots:pane' }
      ]
    },
    { type: 'group', id: 'g-main', panes: ['workspace'], active: 'workspace' }
  ]
}

describe('enforced dock (stacked Bots pane → sessions-zone tab, every boot)', () => {
  beforeEach(() => {
    window.localStorage.clear()
    vi.resetModules()
  })

  afterEach(() => {
    vi.resetModules()
  })

  async function setupTree(initialTree: object, options: { routines?: boolean } = {}) {
    window.localStorage.setItem(TREE_KEY, JSON.stringify(initialTree))

    const tree = await import('@/components/pane-shell/tree/store')
    const model = await import('@/components/pane-shell/tree/model')
    const { registry } = await import('@/contrib/registry')

    registry.register({
      id: 'workspace',
      area: 'panes',
      title: 'chat',
      data: { placement: 'main' },
      render: () => null
    })
    registry.register({
      id: 'sessions',
      area: 'panes',
      title: 'sessions',
      data: { placement: 'left' },
      render: () => null
    })
    registry.register({
      id: 'hermes-bots:pane',
      area: 'panes',
      title: 'Bots',
      data: {
        placement: 'left',
        dock: { pane: 'sessions', pos: 'center', enforce: true }
      },
      render: () => null
    })

    if (options.routines) {
      registry.register({
        id: 'hermes-bots:routines',
        area: 'panes',
        title: 'Cronjobs',
        data: { placement: 'main', dock: { pane: 'workspace', pos: 'right', enforce: true } },
        render: () => null
      })
    }

    return { model, registry, tree }
  }

  async function setup() {
    return setupTree(stackedTree)
  }

  it('re-homes a stacked bots pane into the sessions tab strip, keeping sessions active', async () => {
    const { model, tree } = await setup()

    tree.watchContributedPanes()

    const group = model.findGroupOfPane(tree.$layoutTree.get()!, 'hermes-bots:pane')!

    expect(group.panes).toEqual(['sessions', 'hermes-bots:pane'])
    // Silent like adoption — the enforce must not steal the sessions tab.
    expect(group.active).toBe('sessions')
    // The persisted tree carries the tabbed shape (survives the next boot).
    const persisted = JSON.parse(window.localStorage.getItem(TREE_KEY)!) as { children?: unknown[] }

    expect(JSON.stringify(persisted)).toContain('"panes":["sessions","hermes-bots:pane"]')
  })

  it('keeps a USER-PLACED pane whose persisted group still resolves', async () => {
    window.localStorage.setItem(USER_PLACED_KEY, JSON.stringify(['hermes-bots:pane']))

    const { model, tree } = await setup()

    tree.watchContributedPanes()

    const group = model.findGroupOfPane(tree.$layoutTree.get()!, 'hermes-bots:pane')!

    // The user dragged Bots out of the sessions strip into its own zone.
    // That group still resolves, so boot must not stomp it.
    expect(group.panes).toEqual(['hermes-bots:pane'])
    expect(group.id).toBe('g-bots')
  })

  it('re-homes even when the retired heal token was already burned, and clears the stale ledger', async () => {
    window.localStorage.setItem(LEGACY_HEAL_KEY, JSON.stringify(['hermes-bots:pane:sessions-tab-v1']))

    const { model, tree } = await setup()

    tree.watchContributedPanes()

    const group = model.findGroupOfPane(tree.$layoutTree.get()!, 'hermes-bots:pane')!

    expect(group.panes).toEqual(['sessions', 'hermes-bots:pane'])
    // The one-time-heal ledger is dead state now — importing the store drops it.
    expect(window.localStorage.getItem(LEGACY_HEAL_KEY)).toBeNull()
  })

  it('is idempotent within a boot and does not fight an intra-session drag', async () => {
    const { model, tree, registry } = await setup()

    tree.watchContributedPanes()

    // Sanity: enforced into the strip.
    expect(model.findGroupOfPane(tree.$layoutTree.get()!, 'hermes-bots:pane')!.panes).toContain('sessions')

    // The user drags the pane back out into its own zone below sessions.
    tree.$layoutTree.set(JSON.parse(JSON.stringify(stackedTree)))

    // A later registry mutation re-runs the adoption pass (the enforce's
    // caller) — same boot, so the drag sticks until the next launch.
    registry.register({
      id: 'other',
      area: 'panes',
      title: 'other',
      data: { placement: 'right' },
      render: () => null
    })

    const group = model.findGroupOfPane(tree.$layoutTree.get()!, 'hermes-bots:pane')!

    expect(group.panes).toEqual(['hermes-bots:pane'])
  })

  it('re-homes again on the NEXT boot after a drag persisted the stacked shape', async () => {
    const first = await setup()

    first.tree.watchContributedPanes()
    first.tree.$layoutTree.set(JSON.parse(JSON.stringify(stackedTree)))
    first.tree.persistTree()

    // Simulate the next launch: fresh module graph, persisted stacked tree.
    vi.resetModules()

    const second = await setup()

    second.tree.watchContributedPanes()

    const group = second.model.findGroupOfPane(second.tree.$layoutTree.get()!, 'hermes-bots:pane')!

    expect(group.panes).toEqual(['sessions', 'hermes-bots:pane'])
  })

  it('shows the tab strip when already co-located but hidden with bots active (community "only Bots shows" regression)', async () => {
    // The Aug 2026 field reports: sessions+bots already share one group, the
    // legacy strip flag is set, and bots holds the active tab — the sessions
    // pane exists but is unreachable. The re-home path never runs (nothing to
    // move), so reachability has to come from somewhere else: the migration
    // drops the legacy flag, and a two-pane zone on auto shows its strip.
    const hiddenStackedTree = {
      type: 'split',
      id: 'root',
      orientation: 'row',
      weights: [1, 3],
      children: [
        {
          type: 'group',
          id: 'g-left',
          panes: ['sessions', 'hermes-bots:pane'],
          active: 'hermes-bots:pane',
          headerHidden: true
        },
        { type: 'group', id: 'g-main', panes: ['workspace'], active: 'workspace' }
      ]
    }

    const { model, tree } = await setupTree(hiddenStackedTree)

    tree.watchContributedPanes()

    const group = model.findGroupOfPane(tree.$layoutTree.get()!, 'hermes-bots:pane')!

    // Both panes stay put — but the strip is visible so SESSIONS is reachable
    // again. The active tab is NOT stolen mid-boot.
    expect(group.panes).toEqual(['sessions', 'hermes-bots:pane'])
    expect(tree.tabStripVisibleForGroup(group)).toBe(true)
  })

  it('re-homes an edge-enforced pane stranded in the sessions tab strip', async () => {
    const staleRoutinesTree = {
      type: 'split',
      id: 'root',
      orientation: 'row',
      weights: [1, 3],
      children: [
        {
          type: 'group',
          id: 'g-sessions',
          panes: ['sessions', 'hermes-bots:pane', 'hermes-bots:routines'],
          active: 'hermes-bots:pane'
        },
        { type: 'group', id: 'g-main', panes: ['workspace'], active: 'workspace' }
      ]
    }

    const { model, tree } = await setupTree(staleRoutinesTree, { routines: true })

    tree.watchContributedPanes()

    const botsGroup = model.findGroupOfPane(tree.$layoutTree.get()!, 'hermes-bots:pane')!
    const routinesGroup = model.findGroupOfPane(tree.$layoutTree.get()!, 'hermes-bots:routines')!

    expect(botsGroup.panes).toEqual(['sessions', 'hermes-bots:pane'])
    expect(botsGroup.active).toBe('hermes-bots:pane')
    expect(routinesGroup.panes).toEqual(['hermes-bots:routines'])
    expect(routinesGroup.id).not.toBe(botsGroup.id)
  })

  it('keeps a USER-PLACED edge-enforced pane at a resolvable stacked spot', async () => {
    // Field report: Cronjobs (`pos: 'right'` of workspace, `enforce: true`)
    // was dragged into its own zone under the chat. That placement is
    // recorded and still resolves — boot must not yank it back to a
    // workspace-right split.
    const userStackedRoutinesTree = {
      type: 'split',
      id: 'root',
      orientation: 'row',
      weights: [1, 3],
      children: [
        {
          type: 'group',
          id: 'g-sessions',
          panes: ['sessions', 'hermes-bots:pane'],
          active: 'hermes-bots:pane'
        },
        {
          type: 'split',
          id: 'main-col',
          orientation: 'column',
          weights: [3, 1],
          children: [
            { type: 'group', id: 'g-main', panes: ['workspace'], active: 'workspace' },
            {
              type: 'group',
              id: 'g-routines',
              panes: ['hermes-bots:routines'],
              active: 'hermes-bots:routines'
            }
          ]
        }
      ]
    }

    window.localStorage.setItem(USER_PLACED_KEY, JSON.stringify(['hermes-bots:routines']))

    const { model, tree } = await setupTree(userStackedRoutinesTree, { routines: true })

    tree.watchContributedPanes()

    const routinesGroup = model.findGroupOfPane(tree.$layoutTree.get()!, 'hermes-bots:routines')!

    expect(routinesGroup.panes).toEqual(['hermes-bots:routines'])
    expect(routinesGroup.id).toBe('g-routines')
    expect(tree.$layoutTree.get()).toEqual(userStackedRoutinesTree)
  })

  it('re-homes a USER-PLACED pane whose persisted group no longer resolves', async () => {
    // Anchor destroyed: Cronjobs was user-placed, but its group is gone from
    // the persisted tree (the pane is missing). Enforce + adoption must still
    // guarantee it is present and docked on workspace's right edge.
    const missingRoutinesTree = {
      type: 'split',
      id: 'root',
      orientation: 'row',
      weights: [1, 3],
      children: [
        {
          type: 'group',
          id: 'g-sessions',
          panes: ['sessions', 'hermes-bots:pane'],
          active: 'hermes-bots:pane'
        },
        { type: 'group', id: 'g-main', panes: ['workspace'], active: 'workspace' }
      ]
    }

    window.localStorage.setItem(USER_PLACED_KEY, JSON.stringify(['hermes-bots:routines']))

    const { model, tree } = await setupTree(missingRoutinesTree, { routines: true })

    tree.watchContributedPanes()

    const routinesGroup = model.findGroupOfPane(tree.$layoutTree.get()!, 'hermes-bots:routines')!
    const workspaceGroup = model.findGroupOfPane(tree.$layoutTree.get()!, 'workspace')!

    expect(routinesGroup.panes).toEqual(['hermes-bots:routines'])
    expect(routinesGroup.id).not.toBe(workspaceGroup.id)
    expect(routinesGroup.id).not.toBe('g-sessions')
  })

  it('leaves an edge-enforced pane alone when it already occupies the declared split', async () => {
    const dockedRoutinesTree = {
      type: 'split',
      id: 'root',
      orientation: 'row',
      weights: [1, 3, 1],
      children: [
        {
          type: 'group',
          id: 'g-sessions',
          panes: ['sessions', 'hermes-bots:pane'],
          active: 'hermes-bots:pane'
        },
        { type: 'group', id: 'g-main', panes: ['workspace'], active: 'workspace' },
        {
          type: 'group',
          id: 'g-routines',
          panes: ['hermes-bots:routines'],
          active: 'hermes-bots:routines'
        }
      ]
    }

    const { tree } = await setupTree(dockedRoutinesTree, { routines: true })

    tree.watchContributedPanes()

    expect(tree.$layoutTree.get()).toEqual(dockedRoutinesTree)
  })
})
