import { describe, expect, it } from 'vitest'

import type { EntityKey, LunarCitySnapshot, LunarEntity } from '../model'

import { badgeStatusForEntity, statusBadgeCandidates } from './status-badges'

function entity(key: string, sourceState: string, overrides: Partial<LunarEntity> = {}): LunarEntity {
  return {
    authority: 'authoritative',
    animation: 'idle',
    destination: 'lab',
    identity: { connectionId: 'local', kind: 'profile', profile: key },
    key: key as EntityKey,
    observedAt: 1,
    position: { x: 1, y: 1, z: 1 },
    sourceState,
    ...overrides
  }
}

function snapshot(entities: readonly LunarEntity[]): LunarCitySnapshot {
  return { entities: new Map(entities.map(value => [value.key, value])), observedAt: 1, revision: 1, sources: [] }
}

describe('Lunar City floating status badges', () => {
  it('maps authoritative source states to readable badge classes', () => {
    expect(badgeStatusForEntity(entity('blocked', 'blocked'))).toBe('blocked')
    expect(badgeStatusForEntity(entity('review', 'under_review'))).toBe('review')
    expect(badgeStatusForEntity(entity('delivery', 'delivering'))).toBe('delivering')
    expect(badgeStatusForEntity(entity('waiting', 'queued'))).toBe('waiting')
    expect(badgeStatusForEntity(entity('resting', 'resting'))).toBe('resting')
    expect(badgeStatusForEntity(entity('done', 'completed'))).toBeUndefined()

    for (const authority of ['partial', 'stale', 'unknown'] as const) {
      expect(badgeStatusForEntity(entity('worker', 'running', { authority }))).toBeUndefined()
    }

    for (const state of ['unavailable', 'unrecognized', 'not_working']) {
      expect(badgeStatusForEntity(entity('worker', state))).toBeUndefined()
    }

    expect(badgeStatusForEntity(entity('worker', 'heartbeat'))).toBe('resting')
  })

  it('caps pins, keeps positioned entities only, and prioritizes risk states', () => {
    const hidden = entity('hidden', 'working', { position: undefined })

    const candidates = statusBadgeCandidates(
      snapshot([entity('working', 'working'), hidden, entity('blocked', 'blocked'), entity('review', 'review')]),
      2
    )

    expect(candidates.map(candidate => candidate.entity.key)).toEqual(['blocked', 'review'])
  })
})
