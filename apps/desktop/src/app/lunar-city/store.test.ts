import { describe, expect, it } from 'vitest'

import { entityKey } from './identity'
import { $lunarCitySnapshot, applyLunarDelta, createLunarCitySnapshot } from './store'

const identity = { connectionId: 'local', kind: 'profile' as const, profile: 'worker' }
const key = entityKey(identity)

describe('Lunar City snapshot store', () => {
  it('publishes a frozen copy-on-write delta without mutating the prior snapshot', () => {
    const before = createLunarCitySnapshot()
    $lunarCitySnapshot.set(before)

    const after = applyLunarDelta({
      observedAt: 42,
      removals: [],
      revision: 1,
      sources: [{ authority: 'authoritative', observedAt: 42, source: 'fleet:local' }],
      upserts: [
        {
          animation: 'rest',
          authority: 'authoritative',
          destination: 'garden',
          identity,
          key,
          observedAt: 42,
          presentation: {
            groups: [{ id: 'research-lab', name: 'Research Lab' }],
            metadata: { observedAt: 42, source: 'profiles:local', state: 'fresh' },
            placement: { lodHint: 0, overflow: false, primaryGroupId: 'research-lab', slot: 2 }
          }
        }
      ]
    })

    expect(before).not.toBe(after)
    expect(before.entities.has(key)).toBe(false)
    expect(after.entities.get(key)).toMatchObject({ observedAt: 42 })
    expect(Object.isFrozen(after)).toBe(true)
    expect(Object.isFrozen(after.entities)).toBe(true)
    expect(Object.isFrozen(after.entities.get(key)!)).toBe(true)
    expect(Object.isFrozen(after.entities.get(key)!.presentation)).toBe(true)
    expect(Object.isFrozen(after.entities.get(key)!.presentation!.groups)).toBe(true)
    expect(Object.isFrozen(after.entities.get(key)!.presentation!.groups[0])).toBe(true)
    expect(Object.isFrozen(after.entities.get(key)!.presentation!.metadata)).toBe(true)
    expect(Object.isFrozen(after.entities.get(key)!.presentation!.placement)).toBe(true)
    expect(Object.isFrozen(after.sources[0]!)).toBe(true)
  })

  it('does not republish or regress a snapshot for a duplicate publication revision', () => {
    const current = createLunarCitySnapshot({ revision: 3 })
    $lunarCitySnapshot.set(current)

    const result = applyLunarDelta({ observedAt: 100, removals: [], revision: 3, sources: [], upserts: [] })

    expect(result).toBe(current)
    expect($lunarCitySnapshot.get()).toBe(current)
  })
})
