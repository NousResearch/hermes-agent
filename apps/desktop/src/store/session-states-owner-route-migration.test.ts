import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const TILES_KEY = 'hermes.desktop.sessionTiles.v2'
const OWNER_HINTS_KEY = 'hermes.desktop.sessionOwnerHints.v1'
const ownerRoute = { connectionId: 'remote-a', profile: 'worker', targetProfile: 'worker' }
const otherRoute = { connectionId: 'remote-b', profile: 'worker' }

describe('persisted session tile owner recovery', () => {
  beforeEach(() => {
    window.history.replaceState(null, '', '/')
    window.localStorage.clear()
    vi.resetModules()
  })

  afterEach(() => {
    window.history.replaceState(null, '', '/')
    window.localStorage.clear()
    vi.resetModules()
  })

  it('persists only unambiguous hints and keeps the recovered route after hints expire', async () => {
    window.localStorage.setItem(
      OWNER_HINTS_KEY,
      JSON.stringify([
        ['legacy-session', ownerRoute],
        ['explicit-session', ownerRoute],
        ['ambiguous-session', ownerRoute],
        ['ambiguous-session', otherRoute]
      ])
    )
    window.localStorage.setItem(
      TILES_KEY,
      JSON.stringify({
        default: [
          { storedSessionId: 'legacy-session', workspaceMode: 'sessions' },
          { storedSessionId: 'explicit-session', ownerRoute: otherRoute, workspaceMode: 'sessions' },
          { storedSessionId: 'ambiguous-session', workspaceMode: 'sessions' }
        ]
      })
    )

    const states = await import('./session-states')

    expect(states.sessionTileOwnerRoute('legacy-session')).toEqual(ownerRoute)
    expect(states.sessionTileOwnerRoute('explicit-session')).toEqual(otherRoute)
    expect(states.sessionTileOwnerRoute('ambiguous-session')).toBeUndefined()
    expect(states.openTileGatewayScopes()).toEqual(new Set(['conn:remote-a::worker', 'conn:remote-b::worker']))

    const persisted = JSON.parse(window.localStorage.getItem(TILES_KEY) ?? '{}')

    expect(persisted.default[0].ownerRoute).toEqual(ownerRoute)
    expect(persisted.default[1].ownerRoute).toEqual(otherRoute)
    expect(persisted.default[2].ownerRoute).toBeUndefined()

    window.localStorage.removeItem(OWNER_HINTS_KEY)
    vi.resetModules()

    const restored = await import('./session-states')

    expect(restored.sessionTileOwnerRoute('legacy-session')).toEqual(ownerRoute)
    expect(restored.foregroundSessionScopes()).toContain('conn:remote-a::worker')

    // Hints learned after hydration also route an already mounted legacy tile.
    const sessions = await import('./session')
    sessions.setSessionOwnerHint('late-session', ownerRoute)
    restored.$sessionTiles.set([{ storedSessionId: 'late-session' }])

    expect(restored.sessionTileOwnerRoute('late-session')).toEqual(ownerRoute)
    expect(restored.openTileGatewayScopes()).toEqual(new Set(['conn:remote-a::worker']))
  })

  it('does not rewrite shared tile storage while a pop-out window hydrates', async () => {
    const saved = JSON.stringify({ default: [{ storedSessionId: 'legacy-session', workspaceMode: 'sessions' }] })

    for (const windowKind of ['secondary', 'browser']) {
      vi.resetModules()
      window.history.replaceState(null, '', `/?win=${windowKind}`)
      window.localStorage.setItem(OWNER_HINTS_KEY, JSON.stringify([['legacy-session', ownerRoute]]))
      window.localStorage.setItem(TILES_KEY, saved)

      const states = await import('./session-states')

      expect(states.$sessionTiles.get()).toEqual([])
      expect(window.localStorage.getItem(TILES_KEY)).toBe(saved)
    }
  })
})
