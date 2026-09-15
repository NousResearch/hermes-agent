import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// Honest lock for enforce:true docks: a user placement that would leave the
// declared relationship is refused (tree unchanged) and disclosed once.
// Boot-scoped enforceDockedPanes is unchanged — this file only covers the
// drag/move affordance, not re-home.

function ensureLocalStorage() {
  const current = (globalThis as { localStorage?: Storage }).localStorage

  if (typeof current?.clear === 'function' && typeof current.setItem === 'function') {
    return
  }

  const store = new Map<string, string>()
  const storage: Storage = {
    get length() {
      return store.size
    },
    key: (i: number) => [...store.keys()][i] ?? null,
    getItem: (k: string) => store.get(String(k)) ?? null,
    setItem: (k: string, v: string) => void store.set(String(k), String(v)),
    removeItem: (k: string) => void store.delete(String(k)),
    clear: () => store.clear()
  }

  for (const target of [globalThis, (globalThis as { window?: unknown }).window].filter(Boolean)) {
    Object.defineProperty(target, 'localStorage', {
      configurable: true,
      value: storage,
      writable: true
    })
  }
}

ensureLocalStorage()

const TREE_KEY = 'hermes.desktop.layoutTree.v2'

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

vi.mock('@/store/notifications', () => ({ notify: vi.fn() }))

import { notify } from '@/store/notifications'

describe('enforced-dock drag affordance (honest lock)', () => {
  beforeEach(() => {
    window.localStorage.clear()
    vi.resetModules()
    vi.mocked(notify).mockReset()
  })

  afterEach(() => {
    vi.resetModules()
  })

  async function setupTree(initialTree: object, extra?: { freePane?: boolean }) {
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

    if (extra?.freePane) {
      registry.register({
        id: 'contrib-extra',
        area: 'panes',
        title: 'Extra',
        data: {
          placement: 'left',
          dock: { pane: 'sessions', pos: 'center' }
        },
        render: () => null
      })
    }

    return { model, registry, tree }
  }

  it('does not notify on the silent boot re-home', async () => {
    const { tree } = await setupTree(stackedTree)

    vi.mocked(notify).mockClear()
    tree.watchContributedPanes()

    expect(vi.mocked(notify)).not.toHaveBeenCalled()
  })

  it('refuses moveTreePane of an enforced pane out of its dock and notifies once', async () => {
    const { model, tree } = await setupTree(stackedTree)

    tree.watchContributedPanes()

    const before = tree.$layoutTree.get()!
    const group = model.findGroupOfPane(before, 'hermes-bots:pane')!

    expect(group.panes).toEqual(['sessions', 'hermes-bots:pane'])

    const workspace = model.findGroupOfPane(before, 'workspace')!

    vi.mocked(notify).mockClear()
    tree.moveTreePane('hermes-bots:pane', { groupId: workspace.id, pos: 'right' })

    const after = tree.$layoutTree.get()!

    expect(after).toEqual(before)
    expect(model.findGroupOfPane(after, 'hermes-bots:pane')!.panes).toEqual(['sessions', 'hermes-bots:pane'])
    expect(vi.mocked(notify)).toHaveBeenCalledTimes(1)
  })

  it('CONTROL: a non-enforced contributed pane can still leave its group', async () => {
    const { model, tree } = await setupTree(stackedTree, { freePane: true })

    tree.watchContributedPanes()

    const extraBefore = model.findGroupOfPane(tree.$layoutTree.get()!, 'contrib-extra')!

    expect(extraBefore.panes).toContain('sessions')

    const workspace = model.findGroupOfPane(tree.$layoutTree.get()!, 'workspace')!

    vi.mocked(notify).mockClear()
    tree.moveTreePane('contrib-extra', { groupId: workspace.id, pos: 'right' })

    const extraAfter = model.findGroupOfPane(tree.$layoutTree.get()!, 'contrib-extra')!
    const sessionsAfter = model.findGroupOfPane(tree.$layoutTree.get()!, 'sessions')!

    expect(extraAfter.panes).not.toContain('sessions')
    expect(extraAfter.id).not.toBe(sessionsAfter.id)
    expect(vi.mocked(notify)).not.toHaveBeenCalled()
  })

  it('allows intra-strip reorder of an enforced pane that stays with its anchor', async () => {
    const { model, tree } = await setupTree(stackedTree)

    tree.watchContributedPanes()

    const group = model.findGroupOfPane(tree.$layoutTree.get()!, 'hermes-bots:pane')!

    vi.mocked(notify).mockClear()
    tree.moveTreePane('hermes-bots:pane', { groupId: group.id, pos: 'center', before: 'sessions' })

    const after = model.findGroupOfPane(tree.$layoutTree.get()!, 'hermes-bots:pane')!

    expect(after.panes).toEqual(['hermes-bots:pane', 'sessions'])
    expect(vi.mocked(notify)).not.toHaveBeenCalled()
  })

  it('lets Sessions leave the strip — it is not enforce:true', async () => {
    const { model, tree } = await setupTree(stackedTree)

    tree.watchContributedPanes()

    const workspace = model.findGroupOfPane(tree.$layoutTree.get()!, 'workspace')!

    vi.mocked(notify).mockClear()
    tree.moveTreePane('sessions', { groupId: workspace.id, pos: 'right' })

    const sessionsGroup = model.findGroupOfPane(tree.$layoutTree.get()!, 'sessions')!
    const botsGroup = model.findGroupOfPane(tree.$layoutTree.get()!, 'hermes-bots:pane')!

    expect(sessionsGroup.panes).not.toContain('hermes-bots:pane')
    expect(sessionsGroup.id).not.toBe(botsGroup.id)
    expect(vi.mocked(notify)).not.toHaveBeenCalled()
  })

  it('refuses moveTreePanes of an enforced pane (the drag commit path)', async () => {
    const { model, tree } = await setupTree(stackedTree)

    tree.watchContributedPanes()

    const before = tree.$layoutTree.get()!
    const workspace = model.findGroupOfPane(before, 'workspace')!

    vi.mocked(notify).mockClear()
    tree.moveTreePanes(['hermes-bots:pane'], { groupId: workspace.id, pos: 'right' })

    expect(tree.$layoutTree.get()).toEqual(before)
    expect(vi.mocked(notify)).toHaveBeenCalledTimes(1)
  })

  it('exposes isDockEnforced for the drag-session gate', async () => {
    const { tree } = await setupTree(stackedTree)

    expect(tree.isDockEnforced('hermes-bots:pane')).toBe(true)
    expect(tree.isDockEnforced('sessions')).toBe(false)
    expect(tree.isDockEnforced('missing-pane')).toBe(false)
  })
})
