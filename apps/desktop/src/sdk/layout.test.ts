// Warm the module graph once; each test then re-imports a fresh copy.
import './index'

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// The session-tiling door for plugins: a read-only projection of the pane
// tree, the pinned ids, and layout verbs fenced to chat panes. Before this a
// plugin could open a tile but never see or arrange the layout it landed in.

describe('host.state.sessionLayout / host.layout', () => {
  beforeEach(() => {
    window.localStorage.clear()
    vi.resetModules()
  })

  afterEach(() => {
    vi.resetModules()
  })

  async function setup() {
    const { host } = await import('@/sdk/index')
    const store = await import('@/components/pane-shell/tree/store')
    const model = await import('@/components/pane-shell/tree/model')
    const session = await import('@/store/session')
    const layout = await import('@/store/layout')

    const tree = model.split(
      'row',
      [
        model.group(['workspace'], { id: 'g-main' }),
        model.group(['session-tile:a'], { id: 'g-a', tabStrip: 'always' })
      ],
      [1, 1],
      's-root'
    )

    store.$layoutTree.set(tree)

    return { host, store, session, layout }
  }

  it('projects the pane tree, focused zone and main session', async () => {
    const { host, store, session } = await setup()

    session.$selectedStoredSessionId.set('m')
    store.$activeTreeGroup.set('g-a')

    const l = host.state.sessionLayout.get()

    expect(l.mainStoredSessionId).toBe('m')
    expect(l.focusedGroupId).toBe('g-a')
    expect(l.root).toEqual({
      type: 'split',
      id: 's-root',
      orientation: 'row',
      weights: [1, 1],
      children: [
        { type: 'group', id: 'g-main', panes: ['workspace'], active: 'workspace', minimized: false },
        // internal attrs like tabStrip stay out of the contract
        { type: 'group', id: 'g-a', panes: ['session-tile:a'], active: 'session-tile:a', minimized: false }
      ]
    })
  })

  it('reports a blank main as null', async () => {
    const { host } = await setup()

    expect(host.state.sessionLayout.get().mainStoredSessionId).toBeNull()
  })

  it('mirrors the pinned section ids', async () => {
    const { host, layout } = await setup()

    layout.$pinnedSessionIds.set(['p1', 'p2'])
    expect(host.state.pinnedSessionIds.get()).toEqual(['p1', 'p2'])
  })

  it('resizes and re-orients a split in the live tree', async () => {
    const { host, store } = await setup()

    expect(host.layout.setSplitWeights('s-root', [3, 1])).toBe(true)
    expect((store.$layoutTree.get() as { weights: number[] }).weights).toEqual([3, 1])

    expect(host.layout.setSplitOrientation('s-root', 'column')).toBe(true)
    expect((store.$layoutTree.get() as { orientation: string }).orientation).toBe('column')
  })

  it('rejects bad weights, wrong arity and unknown splits', async () => {
    const { host } = await setup()

    expect(host.layout.setSplitWeights('s-root', [1])).toBe(false)
    expect(host.layout.setSplitWeights('s-root', [1, Number.NaN])).toBe(false)
    expect(host.layout.setSplitWeights('s-root', [1, 0])).toBe(false)
    expect(host.layout.setSplitWeights('nope', [1, 1])).toBe(false)
    expect(host.layout.setSplitOrientation('nope', 'row')).toBe(false)
  })

  it('swaps two chat panes', async () => {
    const { host, store } = await setup()

    expect(host.layout.swapPanes('workspace', 'session-tile:a')).toBe(true)

    const root = store.$layoutTree.get() as { children: { panes: string[] }[] }

    expect(root.children.map(c => c.panes[0])).toEqual(['session-tile:a', 'workspace'])
  })

  it('refuses to touch panes that are not chats', async () => {
    const { host } = await setup()

    expect(host.layout.closePane('terminal')).toBe(false)
    expect(host.layout.closePane('sessions')).toBe(false)
    expect(host.layout.swapPanes('sessions', 'workspace')).toBe(false)
  })
  it('re-tiles the chat area through arrangeSessions, and refuses a spec that misses a chat', async () => {
    const { host, store } = await setup()

    expect(
      host.layout.arrangeSessions({
        orientation: 'column',
        children: [{ pane: 'session-tile:a' }, { pane: 'workspace' }]
      })
    ).toBe(true)

    const root = store.$layoutTree.get()

    expect(root?.type === 'split' && root.orientation).toBe('column')
    expect(host.state.sessionLayout.get().root).toMatchObject({
      type: 'split',
      children: [{ panes: ['session-tile:a'] }, { panes: ['workspace'] }]
    })

    const before = store.$layoutTree.get()

    expect(host.layout.arrangeSessions({ orientation: 'row', children: [{ pane: 'workspace' }] })).toBe(false)
    expect(store.$layoutTree.get()).toBe(before)
  })

  it('opens a fresh chat tile through the wired create path, and reports false until wired', async () => {
    const { host, store } = await setup()

    expect(host.layout.newChatTile()).toBe(false)

    const calls: unknown[] = []

    store.$newSessionTileAction.set(placement => void calls.push(placement))

    expect(host.layout.newChatTile({ pane: 'session-tile:a', pos: 'bottom' })).toBe(true)
    expect(host.layout.newChatTile()).toBe(true)
    expect(host.layout.newChatTile({ pane: 'terminal' })).toBe(false)
    expect(calls).toEqual([
      { pane: 'session-tile:a', pos: 'bottom' },
      { pane: 'workspace', pos: 'right' }
    ])
  })
})
