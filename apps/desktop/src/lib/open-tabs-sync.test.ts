import { describe, expect, it } from 'vitest'

import { decideOpenTabSync, sameOpenTabs } from './open-tabs-sync'

const remote = (revision: number, ids: string[]) => ({
  revision,
  tiles: ids.map(storedSessionId => ({ storedSessionId, dir: 'right' as const })),
  updated_at: '2026-10-01T10:00:00+00:00'
})

describe('decideOpenTabSync', () => {
  it('does not push when the backend cannot be read', () => {
    expect(
      decideOpenTabSync({
        localAppliedRevision: null,
        localTiles: [],
        remote: null
      })
    ).toEqual({ kind: 'stay' })
  })

  it('seeds an empty backend from this device instead of adopting nothing', () => {
    expect(
      decideOpenTabSync({
        localAppliedRevision: null,
        localTiles: [{ storedSessionId: 'local-open' }],
        remote: remote(0, [])
      })
    ).toEqual({ kind: 'seed' })
  })

  it('adopts the other device tabs over a leftover local strip', () => {
    expect(
      decideOpenTabSync({
        localAppliedRevision: null,
        localTiles: [{ storedSessionId: 'stale-on-this-laptop' }],
        remote: remote(4, ['open-on-mini'])
      })
    ).toEqual({
      kind: 'adopt',
      revision: 4,
      tiles: [{ storedSessionId: 'open-on-mini', dir: 'right' }]
    })
  })

  it('adopts an explicit empty strip once this device has synced before', () => {
    const decision = decideOpenTabSync({
      localAppliedRevision: 2,
      localTiles: [{ storedSessionId: 'was-open' }],
      remote: remote(3, [])
    })

    expect(decision).toEqual({ kind: 'adopt', revision: 3, tiles: [] })
  })

  it('pushes only when the revision matches and the local strip changed', () => {
    expect(
      decideOpenTabSync({
        localAppliedRevision: 4,
        localTiles: [{ storedSessionId: 'edited-here' }],
        remote: remote(4, ['open-on-mini'])
      })
    ).toEqual({ kind: 'push-local' })
    expect(
      decideOpenTabSync({
        localAppliedRevision: 4,
        localTiles: [{ storedSessionId: 'open-on-mini', dir: 'right', runtimeId: 'drop-me' } as never],
        remote: remote(4, ['open-on-mini'])
      })
    ).toEqual({ kind: 'noop', revision: 4 })
  })

  it('treats placement-only differences as different strips', () => {
    expect(
      sameOpenTabs(
        [{ storedSessionId: 'a', dir: 'right' }],
        [{ storedSessionId: 'a', dir: 'left' }]
      )
    ).toBe(false)
  })
})
