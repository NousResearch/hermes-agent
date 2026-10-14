// A profile rename keeps its sessions: every localStorage family keyed by the old profile name must move
// to the new one, or the restored tabs dial a backend that no longer exists and 404 forever (#111868).
import { beforeEach, describe, expect, it, vi } from 'vitest'

describe('migrateTilesForProfile', () => {
  beforeEach(() => {
    window.localStorage.clear()
    vi.resetModules()
  })

  it('moves tabs, bot tiles, cached tails, remembered ids and owner hints to the new profile name', async () => {
    const storage = await import('@/lib/storage')
    storage.writeJson('hermes.desktop.sessionTiles.v2', {
      webdesign_bhp: [
        { storedSessionId: 's-1', dir: 'right', ownerRoute: { connectionId: 'local', profile: 'webdesign_bhp' } }
      ],
      __bots_workspace__: [
        {
          storedSessionId: 'bot-1',
          dir: 'right',
          workspaceMode: 'bots',
          ownerRoute: { connectionId: 'local', profile: 'webdesign_bhp', targetProfile: 'webdesign_bhp' }
        },
        {
          storedSessionId: 'bot-2',
          dir: 'right',
          workspaceMode: 'bots',
          ownerRoute: { connectionId: 'local', profile: 'other' }
        }
      ]
    })
    storage.writeJson('hermes.desktop.sessionOwnerHints.v1', [
      ['s-1', { connectionId: 'local', profile: 'webdesign_bhp', targetProfile: 'webdesign_bhp' }]
    ])

    const tails = await import('@/store/transcript-tail-cache')
    tails.saveTranscriptTail('s-1', [{ id: 'm1', parts: [{ text: 'hello', type: 'text' }], role: 'user' } as never], {
      connectionId: 'local',
      profile: 'webdesign_bhp'
    })
    // Same-named profile on a remote connection: not renamed, its tail must stay put.
    tails.saveTranscriptTail('s-9', [{ id: 'm9', parts: [{ text: 'remote', type: 'text' }], role: 'user' } as never], {
      connectionId: 'remote-1',
      profile: 'webdesign_bhp'
    })

    const sessionStore = await import('@/store/session')
    sessionStore.setRememberedSessionId('s-1', 'webdesign_bhp')

    const { migrateTilesForProfile } = await import('@/store/session-states')
    migrateTilesForProfile('webdesign_bhp', 'hutnik-projectmanager')

    type Tile = { ownerRoute?: { profile: string; targetProfile?: string }; storedSessionId: string }
    const tiles = storage.readJson<Record<string, Tile[]>>('hermes.desktop.sessionTiles.v2')
    expect(tiles?.webdesign_bhp).toBeUndefined()
    expect(tiles?.['hutnik-projectmanager']?.map(t => [t.storedSessionId, t.ownerRoute?.profile])).toEqual([
      ['s-1', 'hutnik-projectmanager']
    ])
    expect(tiles?.__bots_workspace__?.map(t => t.ownerRoute)).toEqual([
      { connectionId: 'local', profile: 'hutnik-projectmanager', targetProfile: 'hutnik-projectmanager' },
      { connectionId: 'local', profile: 'other' }
    ])

    expect(tails.loadTranscriptTail('s-1', { connectionId: 'local', profile: 'hutnik-projectmanager' })).toHaveLength(1)
    expect(tails.loadTranscriptTail('s-1', { connectionId: 'local', profile: 'webdesign_bhp' })).toBeNull()
    expect(tails.loadTranscriptTail('s-9', { connectionId: 'remote-1', profile: 'webdesign_bhp' })).toHaveLength(1)
    expect(tails.loadTranscriptTail('s-9', { connectionId: 'remote-1', profile: 'hutnik-projectmanager' })).toBeNull()

    expect(sessionStore.getRememberedSessionId('hutnik-projectmanager')).toBe('s-1')
    expect(sessionStore.getRememberedSessionId('webdesign_bhp')).toBeNull()
    expect(sessionStore.getSessionOwnerHints('s-1')).toEqual([
      { connectionId: 'local', profile: 'hutnik-projectmanager', targetProfile: 'hutnik-projectmanager' }
    ])
  })

  it('leaves owner hints of a same-named profile on another connection alone', async () => {
    const storage = await import('@/lib/storage')
    const remote = { connectionId: 'remote-1', profile: 'webdesign_bhp', targetProfile: 'webdesign_bhp' }
    storage.writeJson('hermes.desktop.sessionOwnerHints.v1', [
      ['s-1', { connectionId: 'local', profile: 'webdesign_bhp', targetProfile: 'webdesign_bhp' }],
      ['s-9', remote]
    ])

    const sessionStore = await import('@/store/session')
    const { migrateTilesForProfile } = await import('@/store/session-states')
    migrateTilesForProfile('webdesign_bhp', 'hutnik-projectmanager')

    expect(sessionStore.getSessionOwnerHints('s-1')).toEqual([
      { connectionId: 'local', profile: 'hutnik-projectmanager', targetProfile: 'hutnik-projectmanager' }
    ])
    // The rename ran on the local backend; the remote profile still has its old name.
    expect(sessionStore.getSessionOwnerHints('s-9')).toEqual([remote])
    expect(storage.readJson<unknown[]>('hermes.desktop.sessionOwnerHints.v1')).toContainEqual(['s-9', remote])
  })
})
