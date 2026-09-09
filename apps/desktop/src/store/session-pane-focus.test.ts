import { afterEach, beforeAll, beforeEach, describe, expect, it } from 'vitest'

import { paneMirror } from '@/app/chat/pane-mirror'
import * as model from '@/components/pane-shell/tree/model'
import * as tree from '@/components/pane-shell/tree/store'
import { registry } from '@/contrib/registry'
import { applyDesktopOverlay } from '@/store/profile-share'
import * as session from '@/store/session'
import * as states from '@/store/session-states'

// These app-lifetime watchers have no unsubscribe API. Install them once in
// Vitest's isolated file graph, not once per case (which accumulates listeners).
beforeAll(() => {
  const disposeWorkspace = registry.register({
    area: 'panes',
    data: { placement: 'main', uncloseable: true },
    id: 'workspace',
    render: () => null,
    title: 'Chat'
  })

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

  return disposeWorkspace
})

function resetState() {
  // Empty the source while the mirror is live so it also disposes its pane.
  states.discardSessionTile('canonical-chat')
  tree.$layoutTree.set(null)
  tree.$activeTreeGroup.set(null)
  tree.$activePresetId.set('default')
  session.$selectedStoredSessionId.set(null)
  session.$lastReadAtBySessionId.set({})
  window.localStorage.clear()
}

function setup() {
  tree.declareDefaultTree(model.group(['workspace'], { active: 'workspace', id: 'main' }))
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
  let ctx: ReturnType<typeof setup>
  const paneId = 'session-tile:canonical-chat'

  beforeEach(() => {
    resetState()
    ctx = setup()
    expect(model.findGroupOfPane(tree.$layoutTree.get()!, 'workspace')?.id).toBe('main')
    expect(states.$sessionTiles.get().map(tile => tile.storedSessionId)).toEqual(['canonical-chat'])
  })

  afterEach(resetState)

  it('re-adopts a saved tab after a profile overlay replaces the layout', () => {
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
