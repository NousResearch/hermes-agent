import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

describe('openSession with the real pane store', () => {
  beforeEach(() => {
    window.localStorage.clear()
    vi.resetModules()
  })

  afterEach(() => {
    vi.resetModules()
  })

  it('preserves main and fronts a session opened with stack intent', async () => {
    const tree = await import('@/components/pane-shell/tree/store')
    const model = await import('@/components/pane-shell/tree/model')
    const { registry } = await import('@/contrib/registry')
    const session = await import('@/store/session')
    const { watchSessionTiles } = await import('./chat/session-tile')
    const { openSession } = await import('./open-session')

    registry.register({
      area: 'panes',
      data: { placement: 'main', uncloseable: true },
      id: 'workspace',
      render: () => null,
      title: 'chat'
    })

    watchSessionTiles()
    tree.watchContributedPanes()
    session.$selectedStoredSessionId.set('primary')
    tree.declareDefaultTree(model.group(['workspace'], { active: 'workspace', id: 'grp-main' }))

    openSession('target', vi.fn(), 'stack')

    expect(session.$selectedStoredSessionId.get()).toBe('primary')
    expect(model.findGroupOfPane(tree.$layoutTree.get()!, 'session-tile:target')?.active).toBe('session-tile:target')
  })
})
