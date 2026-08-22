import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const STORAGE_KEY = 'hermes.desktop.sessionTiles.v2'

const storedTile = (storedSessionId: string, profile: string) => ({
  profile,
  storedSessionId
})

describe('profile-scoped session tile visibility', () => {
  beforeEach(() => {
    vi.resetModules()
    localStorage.clear()
  })

  afterEach(() => {
    localStorage.clear()
    vi.resetModules()
  })

  it('restores each owner set and aggregates every owner only in explicit All Profiles', async () => {
    localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({
        default: [storedTile('default-tab', 'default')],
        work: [storedTile('work-tab', 'work')]
      })
    )

    const profile = await import('./profile')
    const states = await import('./session-states')

    expect(states.$sessionTiles.get().map(tile => tile.storedSessionId)).toEqual(['default-tab'])

    profile.$activeGatewayProfile.set('work')
    expect(states.$sessionTiles.get().map(tile => tile.storedSessionId)).toEqual(['work-tab'])

    profile.$showAllProfiles.set(true)
    expect(states.$sessionTiles.get().map(tile => tile.storedSessionId)).toEqual(['default-tab', 'work-tab'])

    profile.$showAllProfiles.set(false)
    expect(states.$sessionTiles.get().map(tile => tile.storedSessionId)).toEqual(['work-tab'])
  })

  it('preserves the owner when a foreign All Profiles tab is closed and reopened', async () => {
    localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({
        default: [storedTile('default-tab', 'default')],
        work: [storedTile('work-tab', 'work')]
      })
    )

    const profile = await import('./profile')
    const states = await import('./session-states')

    profile.$showAllProfiles.set(true)
    states.closeSessionTile('work-tab')
    states.reopenLastClosedTile()

    expect(states.$sessionTiles.get().find(tile => tile.storedSessionId === 'work-tab')?.profile).toBe('work')
  })
})
