import { afterEach, describe, expect, it } from 'vitest'

// The completed-unread dot is keyed on the FOCUSED session, not the selected
// one. A tile is never $selectedStoredSessionId, so keying either half on the
// selection left a tiled session's dot green with no way to clear it.

describe('completed-unread dot follows the focused session', () => {
  const disposers: (() => void)[] = []

  afterEach(() => {
    // Undo the registrations and store writes this file makes, so the next
    // test starts clean without paying to rebuild the module graph.
    while (disposers.length > 0) {
      disposers.pop()?.()
    }
  })

  async function setup() {
    const tree = await import('@/components/pane-shell/tree/store')
    const model = await import('@/components/pane-shell/tree/model')
    const { registry } = await import('@/contrib/registry')
    const { createClientSessionState } = await import('@/lib/chat-runtime')
    const session = await import('./session')
    const states = await import('./session-states')

    // `declareDefaultTree` only seeds `$layoutTree` when it is empty, so the
    // layout has to be cleared here or the second test would adopt the first
    // test's tree instead of the one it declares.
    tree.$layoutTree.set(null)
    tree.$activeTreeGroup.set(null)

    for (const id of ['workspace', 'session-tile:tiled']) {
      disposers.push(
        registry.register({
          area: 'panes',
          data:
            id === 'workspace' ? { placement: 'main', uncloseable: true } : { placement: 'main' },
          id,
          render: () => null,
          title: id
        })
      )
    }

    // The workspace holds the primary chat, a second zone holds the tile.
    tree.declareDefaultTree(
      model.split('row', [
        model.group(['workspace'], { active: 'workspace', id: 'grp-main' }),
        model.group(['session-tile:tiled'], { active: 'session-tile:tiled', id: 'grp-tile' })
      ])
    )

    session.$unreadFinishedSessionIds.set([])
    session.$selectedStoredSessionId.set('primary')

    const finishTurn = (storedSessionId: string) => {
      const working = { ...createClientSessionState(null), busy: true, storedSessionId }
      states.publishSessionState(`rt-${storedSessionId}`, working)
      states.publishSessionState(`rt-${storedSessionId}`, { ...working, busy: false })
    }

    return { finishTurn, session, tree }
  }

  it('clears the dot when an already-open tile is fronted', async () => {
    const { finishTurn, session, tree } = await setup()

    tree.noteActiveTreeGroup('grp-main')
    finishTurn('tiled')
    expect(session.$unreadFinishedSessionIds.get()).toEqual(['tiled'])

    // Fronting the tile is what a tab click does. Before the fix nothing on
    // this path cleared the marker, so the dot stayed green.
    tree.noteActiveTreeGroup('grp-tile')
    expect(session.$unreadFinishedSessionIds.get()).toEqual([])
  })

  it('never marks a tile that finishes while it is the focused one', async () => {
    const { finishTurn, session, tree } = await setup()

    tree.noteActiveTreeGroup('grp-tile')
    finishTurn('tiled')

    expect(session.$unreadFinishedSessionIds.get()).toEqual([])
  })

  it('marks the primary session when a tile has focus', async () => {
    const { finishTurn, session, tree } = await setup()

    tree.noteActiveTreeGroup('grp-tile')
    finishTurn('primary')

    expect(session.$unreadFinishedSessionIds.get()).toEqual(['primary'])
  })
})
