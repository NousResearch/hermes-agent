import { beforeEach, describe, expect, it, vi } from 'vitest'

async function setup() {
  const tree = await import('@/components/pane-shell/tree/store')
  const model = await import('@/components/pane-shell/tree/model')
  const { registry } = await import('@/contrib/registry')
  const session = await import('@/store/session')
  const states = await import('@/store/session-states')
  const { paneMirror } = await import('@/app/chat/pane-mirror')

  registry.register({
    area: 'panes',
    data: { placement: 'main', uncloseable: true },
    id: 'workspace',
    render: () => null,
    title: 'Chat'
  })
  tree.declareDefaultTree(model.group(['workspace'], { active: 'workspace', id: 'main' }))
  tree.watchContributedPanes()
  paneMirror({
    source: states.$sessionTiles,
    key: tile => tile.storedSessionId,
    prefix: 'session-tile',
    dir: () => 'center',
    minWidth: '20rem',
    title: id => id,
    render: () => null,
    close: states.closeSessionTile
  })()
  session.$selectedStoredSessionId.set('previous-chat')

  const scope = {
    ownerRoute: { connectionId: 'remote-a', mode: 'remote' as const, profile: 'writer' },
    workspaceMode: 'bots' as const,
    workspaceOwnerKey: 'remote-a::writer',
    workspaceTabTitle: 'Bot Chat'
  }

  states.openSessionTile('canonical-chat', 'center', 'workspace', undefined, scope)

  return { model, scope, session, states, tree }
}

describe('focusing a saved Bot Chat requires a visible pane', () => {
  let ctx: Awaited<ReturnType<typeof setup>>
  const paneId = 'session-tile:canonical-chat'

  beforeEach(async () => {
    window.localStorage.clear()
    vi.resetModules()
    ctx = await setup()
  })

  it('re-adopts a saved tab after a profile overlay replaces the layout', async () => {
    const { applyDesktopOverlay } = await import('@/store/profile-share')
    const { model, scope, states, tree } = ctx
    const saved = states.$sessionTiles.get()
    applyDesktopOverlay('imported-profile', {
      version: 1,
      layoutTree: model.group(['workspace'], { active: 'workspace', id: 'imported-main' })
    })
    expect(model.findGroupOfPane(tree.$layoutTree.get()!, paneId)).toBeNull()

    expect(states.focusWorkspaceOwnerSessionTile(scope.workspaceOwnerKey, undefined, ['canonical-chat'])).toBe(
      'canonical-chat'
    )
    expect(tree.isPaneVisible(paneId)).toBe(true)
    expect(tree.$activeTreeGroup.get()).toBe('imported-main')
    expect(states.$sessionTiles.get()).toEqual(saved)
    expect(states.sessionTileOwnerRoute('canonical-chat')).toEqual(scope.ownerRoute)
  })

  it('reports a miss through both helpers if the layout cannot place the saved tab', () => {
    const { scope, session, states, tree } = ctx
    tree.$layoutTree.set(null)

    expect(states.focusOpenSession('canonical-chat', scope)).toBeNull()
    expect(states.focusWorkspaceOwnerSessionTile(scope.workspaceOwnerKey, undefined, ['canonical-chat'])).toBeNull()
    expect(session.$selectedStoredSessionId.get()).toBe('previous-chat')
    expect(states.$sessionTiles.get().map(tile => tile.storedSessionId)).toEqual(['canonical-chat'])
  })
})
