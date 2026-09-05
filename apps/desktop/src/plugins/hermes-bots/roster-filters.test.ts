/**
 * The roster's filters are a preference, not a gesture: what you picked must
 * still be picked after a reload. Asserts the contract (a set survives a fresh
 * module load), not the storage key's spelling.
 */

import { beforeEach, describe, expect, it, vi } from 'vitest'

async function freshModule() {
  vi.resetModules()

  return import('./roster-filters')
}

describe('roster filters', () => {
  beforeEach(() => {
    localStorage.clear()
    vi.resetModules()
  })

  it('defaults to unfiltered', async () => {
    const m = await freshModule()

    expect(m.$rosterKindFilter.get()).toBe('all')
    expect(m.$rosterActivityFilter.get()).toBe('all')
    expect(m.$rosterGatewayFilter.get()).toBe('all')
  })

  it('survives a reload', async () => {
    const first = await freshModule()

    first.$rosterKindFilter.set('groups')
    first.$rosterActivityFilter.set('recent')
    first.$rosterGatewayFilter.set('conn-42')

    const reloaded = await freshModule()

    expect(reloaded.$rosterKindFilter.get()).toBe('groups')
    expect(reloaded.$rosterActivityFilter.get()).toBe('recent')
    expect(reloaded.$rosterGatewayFilter.get()).toBe('conn-42')
  })

  it('falls back to all when a stored value is no longer a valid option', async () => {
    const first = await freshModule()

    first.$rosterKindFilter.set('groups')

    // A stale key from an older build, or a hand-edited one.
    const key = Object.keys(localStorage).find(k => k.includes('KindFilter'))

    expect(key).toBeDefined()
    localStorage.setItem(key as string, 'nonsense')

    const reloaded = await freshModule()

    expect(reloaded.$rosterKindFilter.get()).toBe('all')
  })

  it('reset clears all three', async () => {
    const m = await freshModule()

    m.$rosterKindFilter.set('bots')
    m.$rosterActivityFilter.set('older')
    m.$rosterGatewayFilter.set('conn-7')

    m.resetRosterFilters()

    expect(m.$rosterKindFilter.get()).toBe('all')
    expect(m.$rosterActivityFilter.get()).toBe('all')
    expect(m.$rosterGatewayFilter.get()).toBe('all')
  })
})
