import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { HermesConnection } from '@/global'

import { $activeGatewayProfile } from './profile'
import { $selectedStoredSessionId, setConnection } from './session'
import { $sessionTiles, openSessionTile } from './session-states'

const profile = 'connection-scope-regression'
const local = {
  baseUrl: 'http://127.0.0.1:8000',
  connectionId: 'local',
  mode: 'local',
  profile
} as unknown as HermesConnection
const remote = {
  baseUrl: 'https://homelab.example:8443',
  connectionId: 'homelab',
  mode: 'remote',
  profile
} as unknown as HermesConnection
const otherRemote = {
  baseUrl: 'https://other.example:8443',
  connectionId: 'other-remote',
  mode: 'remote',
  profile
} as unknown as HermesConnection

beforeEach(() => {
  window.localStorage.clear()
  setConnection(local)
  $activeGatewayProfile.set(profile)
  $selectedStoredSessionId.set(null)
  $sessionTiles.set([])
})

describe('session tiles across registered backend switches (#120106)', () => {
  it('keeps each same-named profile on its own connection and restores its tabs on return', async () => {
    openSessionTile('stored-local', 'right', undefined, undefined, {
      workspaceMode: 'sessions',
      ownerRoute: { connectionId: 'local', profile }
    })
    expect($sessionTiles.get().map(tile => tile.storedSessionId)).toEqual(['stored-local'])

    setConnection(remote)
    expect($sessionTiles.get().map(tile => tile.storedSessionId)).toEqual([])

    openSessionTile('stored-remote', 'right', undefined, undefined, {
      workspaceMode: 'sessions',
      ownerRoute: { connectionId: 'homelab', profile }
    })
    expect($sessionTiles.get().map(tile => tile.storedSessionId)).toEqual(['stored-remote'])

    // A disconnect is not a source change; another remote with the same
    // profile is a different workspace.
    setConnection(null)
    expect($sessionTiles.get().map(tile => tile.storedSessionId)).toEqual(['stored-remote'])
    setConnection(otherRemote)
    expect($sessionTiles.get().map(tile => tile.storedSessionId)).toEqual([])
    setConnection(remote)
    expect($sessionTiles.get().map(tile => tile.storedSessionId)).toEqual(['stored-remote'])

    setConnection(local)
    expect($sessionTiles.get().map(tile => tile.storedSessionId)).toEqual(['stored-local'])
    expect($sessionTiles.get()[0]?.ownerRoute?.connectionId).toBe('local')

    setConnection(remote)
    expect($sessionTiles.get().map(tile => tile.storedSessionId)).toEqual(['stored-remote'])
    expect($sessionTiles.get()[0]?.ownerRoute?.connectionId).toBe('homelab')

    // A restart must keep the composite key; profile-only migration would
    // silently fold remote tabs back into the local bucket.
    const persisted = JSON.parse(window.localStorage.getItem('hermes.desktop.sessionTiles.v2') || '{}')
    expect(persisted[profile]?.map((tile: { storedSessionId: string }) => tile.storedSessionId)).toEqual([
      'stored-local'
    ])
    expect(
      persisted[`conn:homelab::${profile}`]?.map((tile: { storedSessionId: string }) => tile.storedSessionId)
    ).toEqual(['stored-remote'])

    vi.resetModules()
    const reloadedSession = await import('./session')
    const reloadedProfile = await import('./profile')
    const reloadedTiles = await import('./session-states')
    reloadedProfile.$activeGatewayProfile.set(profile)
    reloadedSession.setConnection(remote)
    expect(reloadedTiles.$sessionTiles.get().map(tile => tile.storedSessionId)).toEqual(['stored-remote'])
    reloadedSession.setConnection(local)
    expect(reloadedTiles.$sessionTiles.get().map(tile => tile.storedSessionId)).toEqual(['stored-local'])

    // A source-scoped deletion must leave the same-named local profile alone.
    reloadedTiles.dropTilesForProfile(profile, { connectionId: 'homelab', profile })
    reloadedSession.setConnection(remote)
    expect(reloadedTiles.$sessionTiles.get()).toEqual([])
    reloadedSession.setConnection(local)
    expect(reloadedTiles.$sessionTiles.get().map(tile => tile.storedSessionId)).toEqual(['stored-local'])
  })

  it('migrates legacy remote-owned tabs out of a profile-only bucket', async () => {
    window.localStorage.setItem(
      'hermes.desktop.sessionTiles.v2',
      JSON.stringify({
        [profile]: [
          { dir: 'right', ownerRoute: { connectionId: 'local', profile }, storedSessionId: 'legacy-local' },
          { dir: 'right', ownerRoute: { connectionId: 'homelab', profile }, storedSessionId: 'legacy-remote' },
          { dir: 'right', storedSessionId: 'legacy-unknown-owner' }
        ]
      })
    )
    vi.resetModules()
    const session = await import('./session')
    const gatewayProfile = await import('./profile')
    const tiles = await import('./session-states')
    gatewayProfile.$activeGatewayProfile.set(profile)

    session.setConnection(remote)
    expect(tiles.$sessionTiles.get().map(tile => tile.storedSessionId)).toEqual(['legacy-remote'])
    session.setConnection(local)
    expect(tiles.$sessionTiles.get().map(tile => tile.storedSessionId)).toEqual([
      'legacy-local',
      'legacy-unknown-owner'
    ])
  })
})
