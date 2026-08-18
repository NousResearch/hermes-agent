import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// One-time dock heals: a pane already adopted under an OLD dock hint (Bot
// Mode's Bots pane split BELOW sessions) re-homes onto its new center anchor
// exactly once — persisted layouts otherwise pin the stale arrangement
// forever, because adoption only ever places panes MISSING from the tree.
// Guards under test: user-placed layouts are never touched, and a burned
// token never re-fights the user across later adoption passes.

const TREE_KEY = 'hermes.desktop.layoutTree.v2'
const USER_PLACED_KEY = 'hermes.desktop.userPlacedPanes.v1'

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

describe('one-time dock heal (stacked Bots pane → sessions-zone tab)', () => {
  beforeEach(() => {
    window.localStorage.clear()
    vi.resetModules()
  })

  afterEach(() => {
    vi.resetModules()
  })

  async function setup() {
    window.localStorage.setItem(TREE_KEY, JSON.stringify(stackedTree))

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
        dock: { pane: 'sessions', pos: 'center', heal: 'sessions-tab-v1' }
      },
      render: () => null
    })

    return { model, registry, tree }
  }

  it('re-homes a stacked bots pane into the sessions tab strip, keeping sessions active', async () => {
    const { model, tree } = await setup()

    tree.watchContributedPanes()

    const group = model.findGroupOfPane(tree.$layoutTree.get()!, 'hermes-bots:pane')!

    expect(group.panes).toEqual(['sessions', 'hermes-bots:pane'])
    // Silent like adoption — the heal must not steal the sessions tab.
    expect(group.active).toBe('sessions')
    // The persisted tree carries the healed shape (survives the next boot).
    const persisted = JSON.parse(window.localStorage.getItem(TREE_KEY)!) as { children?: unknown[] }

    expect(JSON.stringify(persisted)).toContain('"panes":["sessions","hermes-bots:pane"]')
  })

  it('never touches a layout the user placed the pane in themselves', async () => {
    window.localStorage.setItem(USER_PLACED_KEY, JSON.stringify(['hermes-bots:pane']))

    const { model, tree } = await setup()

    tree.watchContributedPanes()

    const group = model.findGroupOfPane(tree.$layoutTree.get()!, 'hermes-bots:pane')!

    // Still its own zone below sessions — the user's spot wins.
    expect(group.panes).toEqual(['hermes-bots:pane'])
  })

  it('burns its token once: a user who re-stacks the pane afterward is not fought', async () => {
    const { model, tree, registry } = await setup()

    tree.watchContributedPanes()

    // Sanity: healed.
    expect(model.findGroupOfPane(tree.$layoutTree.get()!, 'hermes-bots:pane')!.panes).toContain('sessions')

    // The user drags the pane back out into its own zone below sessions.
    tree.$layoutTree.set(JSON.parse(JSON.stringify(stackedTree)))

    // A later registry mutation re-runs the adoption pass (the heal's caller).
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
})
