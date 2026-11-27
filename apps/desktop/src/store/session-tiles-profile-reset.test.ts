/**
 * Regression: a layout reset must clear EVERY profile's persisted session
 * tiles, not just the live one.
 *
 * Tiles are stored per profile (`hermes.desktop.sessionTiles.v2` is a map keyed
 * by profile). Before `clearInactiveProfileTiles`, a reset only emptied the
 * active profile's entry, so the moment the gateway swapped — a bot click, the
 * profile rail — the other profile's saved tiles re-adopted at their stored
 * edges and the window went back to the pre-reset split. The reset visibly
 * undid itself one profile later.
 */

import { beforeEach, describe, expect, it, vi } from 'vitest'

import type * as ProfileStore from '@/store/profile'

const TILES_KEY = 'hermes.desktop.sessionTiles.v2'

describe('layout reset clears tiles across profiles', () => {
  beforeEach(() => {
    window.localStorage.clear()
    vi.resetModules()
  })

  async function loadWithProfile(active: string) {
    vi.doMock('@/store/profile', async () => {
      const actual = await vi.importActual<typeof ProfileStore>('@/store/profile')
      const { atom } = await import('nanostores')

      return { ...actual, $activeGatewayProfile: atom(active) }
    })

    return import('@/store/session-states')
  }

  it('drops inactive profiles from storage while leaving the live one alone', async () => {
    window.localStorage.setItem(
      TILES_KEY,
      JSON.stringify({
        curolab: [{ dir: 'right', storedSessionId: 'curolab-1' }],
        ops: [{ dir: 'right', storedSessionId: 'ops-1' }],
        plan: [{ dir: 'right', storedSessionId: 'plan-1' }, { dir: 'bottom', storedSessionId: 'plan-2' }]
      })
    )

    const states = await loadWithProfile('ops')

    // The live profile hydrates its own tiles; the others sit in storage.
    expect(states.$sessionTiles.get().map(t => t.storedSessionId)).toEqual(['ops-1'])

    states.clearInactiveProfileTiles()

    const stored = JSON.parse(window.localStorage.getItem(TILES_KEY) ?? '{}')

    expect(Object.keys(stored)).toEqual(['ops'])
    expect(stored.plan).toBeUndefined()
    expect(stored.curolab).toBeUndefined()
    // The live atom is untouched — the normal reset handler stacks these.
    expect(states.$sessionTiles.get().map(t => t.storedSessionId)).toEqual(['ops-1'])
  })

  it('a profile swap after the clear surfaces nothing (the reset actually held)', async () => {
    window.localStorage.setItem(
      TILES_KEY,
      JSON.stringify({
        ops: [{ dir: 'right', storedSessionId: 'ops-1' }],
        plan: [{ dir: 'right', storedSessionId: 'plan-1' }]
      })
    )

    const opsStates = await loadWithProfile('ops')
    opsStates.clearInactiveProfileTiles()

    // Re-import as if the gateway swapped to `plan` (a bot click).
    vi.resetModules()
    const planStates = await loadWithProfile('plan')

    expect(planStates.$sessionTiles.get()).toEqual([])
  })
})
