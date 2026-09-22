import { describe, expect, it, vi } from 'vitest'

import type { DesktopAgentRoster } from '@/global'

import type { LunarEntity } from '../model'

import {
  assignStablePlacementSlots,
  createBotRosterPlacementState,
  enrichBotRosterEntities
} from './bot-roster-details'
import { normalizeRoster } from './fleet'

function roster(profiles: readonly string[], connectionId = 'local'): DesktopAgentRoster {
  return {
    agents: profiles.map(profile => ({
      connectionId,
      connectionKind: 'local',
      connectionLabel: connectionId === 'local' ? 'This device' : 'Hermes Desktop',
      handle: `@${profile}-${connectionId}`,
      profile
    })),
    sources: [
      {
        connectionId,
        kind: 'local',
        label: connectionId === 'local' ? 'This device' : 'Hermes Desktop',
        reachable: connectionId === 'local'
      }
    ]
  }
}

function entities(source: DesktopAgentRoster): readonly LunarEntity[] {
  return normalizeRoster(source, { observedAt: 42 }).entities
}

describe('complete Hermes Bots roster details', () => {
  it('retains all 91 fleet profiles from the 113-item live UI while carrying its 19 groups as memberships', async () => {
    const localProfiles = Array.from({ length: 90 }, (_, index) => `worker-${index}`)
    const local = roster(localProfiles)
    const unavailable = roster(['test-contract-steward'], 'desktop-source')

    const source: DesktopAgentRoster = {
      agents: [...local.agents, ...unavailable.agents],
      sources: [...local.sources, ...unavailable.sources]
    }

    const liveGroupRooms = Object.fromEntries(
      Array.from({ length: 19 }, (_, index) => [
        `id:group-${index}`,
        {
          members: [{ connectionId: 'local', name: 'worker-0' }],
          name: `Live group ${index}`,
          roomId: `group-${index}`
        }
      ])
    )

    const request = vi.fn(async (connectionId: string, profile: string) => ({
      profiles:
        connectionId === 'local'
          ? localProfiles.map((name, index) => ({
              name,
              ...(index === 0 ? { ui_meta: { 'hermes-bots-groups': { rooms: liveGroupRooms, version: 3 } } } : {})
            }))
          : [{ name: profile }]
    }))

    const result = await enrichBotRosterEntities(source, entities(source), request)

    expect(result).toHaveLength(91)
    expect(new Set(result.map(entity => entity.key)).size).toBe(91)
    expect(result.find(entity => entity.identity.profile === 'worker-0')?.presentation?.groups).toHaveLength(19)
    expect(request).toHaveBeenCalledTimes(2)
    expect(request).toHaveBeenCalledWith('desktop-source', 'test-contract-steward')
    expect(request).toHaveBeenCalledWith('local', 'worker-0')
    expect(result.find(entity => entity.identity.connectionId === 'desktop-source')).toMatchObject({
      authority: 'stale',
      destination: 'unavailable',
      presentation: { profileHandle: '@test-contract-steward-desktop-source', sourceLabel: 'Hermes Desktop' }
    })
  })

  it('uses configured title precedence, retains every direct and projected group, and maps the primary group explicitly', async () => {
    const source = roster(['scientific-validator'])

    const request = vi.fn(async () => ({
      profiles: [
        {
          display_name: 'Display fallback',
          name: 'scientific-validator',
          title: 'Core title',
          ui_meta: {
            'hermes-bots': {
              group: 'Research Lab',
              groups: ['Engineering Guild', 'Research Lab', 'Research Review Board'],
              title: 'Scientific Validator'
            },
            'hermes-bots-groups': {
              rooms: {
                'id:engineering': {
                  members: [{ connectionId: 'local', name: 'scientific-validator' }],
                  name: 'Engineering Guild',
                  roomId: 'engineering'
                }
              },
              version: 3
            }
          }
        }
      ]
    }))

    const [entity] = await enrichBotRosterEntities(source, entities(source), request)

    expect(entity?.destination).toBe('project')
    expect(entity?.presentation).toMatchObject({
      configuredTitle: 'Scientific Validator',
      groups: [
        { id: 'engineering', name: 'Engineering Guild' },
        { id: 'research-lab', name: 'Research Lab' },
        { id: 'research-review-board', name: 'Research Review Board' }
      ],
      placement: { overflow: false, primaryGroupId: 'engineering' },
      profileHandle: '@scientific-validator-local',
      sourceLabel: 'This device'
    })
    expect(entity?.position).toBeDefined()
  })

  it('keeps same-named profiles on separate sources distinct and never uses another source metadata', async () => {
    const left = roster(['steward'], 'local')
    const right = roster(['steward'], 'desktop-source')
    const source = { agents: [...left.agents, ...right.agents], sources: [...left.sources, ...right.sources] }

    const request = vi.fn(async (connectionId: string) => ({
      profiles: [
        {
          name: 'steward',
          ui_meta: { 'hermes-bots': { title: connectionId === 'local' ? 'Ready Steward' : 'Offline Steward' } }
        }
      ]
    }))

    const result = await enrichBotRosterEntities(source, entities(source), request)

    expect(result.map(entity => entity.presentation?.configuredTitle)).toEqual(['Offline Steward', 'Ready Steward'])
    expect(new Set(result.map(entity => entity.key)).size).toBe(2)
  })

  it('shape-guards malformed metadata and metadata failure without dropping or authorizing a fleet row', async () => {
    const source = roster(['alpha', 'beta'])

    const malformed = await enrichBotRosterEntities(source, entities(source), async () => ({
      profiles: [
        {
          name: 'alpha',
          ui_meta: {
            'hermes-bots': {
              groups: ['Research Lab', null, 7, 'x'.repeat(500)],
              title: { untrusted: true }
            },
            'hermes-bots-groups': { rooms: 'not-an-object' }
          }
        }
      ]
    }))

    const failed = await enrichBotRosterEntities(source, entities(source), async () => {
      throw new Error('offline')
    })

    expect(malformed).toHaveLength(2)
    expect(malformed[0]?.presentation?.configuredTitle).toBeUndefined()
    expect(malformed[0]?.presentation?.groups).toEqual([{ id: 'research-lab', name: 'Research Lab' }])
    expect(failed).toHaveLength(2)
    expect(failed.every(entity => entity.authority === 'authoritative')).toBe(true)
    expect(failed.every(entity => entity.destination === 'garden')).toBe(true)
    expect(failed.every(entity => entity.presentation?.metadata.state === 'unavailable')).toBe(true)
    expect(failed.find(entity => entity.identity.profile === 'beta')?.presentation?.groups).toEqual([])
  })

  it('assigns collision-safe stable slots independent of roster ordering and flags bounded overflow for aggregate LOD', async () => {
    const profiles = Array.from({ length: 40 }, (_, index) => `engineer-${index}`)
    const source = roster(profiles)

    const richRows = profiles.map(name => ({
      name,
      ui_meta: { 'hermes-bots': { groups: ['Engineering Guild'] } }
    }))

    const forward = await enrichBotRosterEntities(source, entities(source), async () => ({ profiles: richRows }))
    const reversedRoster = { ...source, agents: [...source.agents].reverse() }

    const reverse = await enrichBotRosterEntities(reversedRoster, entities(reversedRoster), async () => ({
      profiles: [...richRows].reverse()
    }))

    const forwardByKey = new Map(forward.map(entity => [entity.key, entity]))

    expect(new Set(forward.map(entity => JSON.stringify(entity.position))).size).toBe(40)
    expect(
      reverse.every(
        entity => JSON.stringify(entity.position) === JSON.stringify(forwardByKey.get(entity.key)?.position)
      )
    ).toBe(true)
    expect(forward.filter(entity => entity.presentation?.placement.overflow)).toHaveLength(16)
    expect(forward.filter(entity => entity.presentation?.placement.lodHint === 1)).toHaveLength(16)
  })

  it('keeps every bounded placement unique across capped-ring and former 256-slot collision boundaries', async () => {
    const profiles = Array.from({ length: 320 }, (_, index) => `worker-${index.toString().padStart(3, '0')}`)
    const source = roster(profiles)

    const result = await enrichBotRosterEntities(source, entities(source), async () => ({
      profiles: profiles.map(name => ({ name }))
    }))

    expect(new Set(result.map(entity => entity.presentation?.placement.slot)).size).toBe(320)
    expect(new Set(result.map(entity => JSON.stringify(entity.position))).size).toBe(320)
    expect(result[228]?.position).not.toEqual(result[240]?.position)
    expect(result[255]?.position).not.toEqual(result[256]?.position)
    expect(result[319]?.presentation?.placement.slot).toBeDefined()
  })

  it('keeps exact-identity placement stable when an earlier key is inserted and bounds a maximum source roster', async () => {
    const originalProfiles = Array.from(
      { length: 320 },
      (_, index) => `worker-${(index + 100).toString().padStart(4, '0')}`
    )

    const originalRoster = roster(originalProfiles)

    const original = await enrichBotRosterEntities(originalRoster, entities(originalRoster), async () => ({
      profiles: originalProfiles.map(name => ({ name }))
    }))

    const insertedProfiles = ['aaa-earlier-worker', ...originalProfiles]
    const insertedRoster = roster(insertedProfiles)

    const inserted = await enrichBotRosterEntities(insertedRoster, entities(insertedRoster), async () => ({
      profiles: insertedProfiles.map(name => ({ name }))
    }))

    const originalByKey = new Map(original.map(entity => [entity.key, entity]))

    expect(
      inserted
        .filter(entity => entity.identity.profile !== 'aaa-earlier-worker')
        .every(entity => JSON.stringify(entity.position) === JSON.stringify(originalByKey.get(entity.key)?.position))
    ).toBe(true)

    const maximumProfiles = Array.from({ length: 2048 }, (_, index) => `maximum-${index.toString().padStart(4, '0')}`)
    const maximumRoster = roster(maximumProfiles)

    const maximum = await enrichBotRosterEntities(maximumRoster, entities(maximumRoster), async () => ({
      profiles: maximumProfiles.map(name => ({ name }))
    }))

    expect(new Set(maximum.map(entity => JSON.stringify(entity.position))).size).toBe(2048)
    expect(
      maximum.every(
        entity =>
          entity.position !== undefined &&
          entity.position.x >= -60 &&
          entity.position.x <= 60 &&
          entity.position.z >= -60 &&
          entity.position.z <= 60
      )
    ).toBe(true)
  })

  it('keeps the concrete signed-step collision pair unique and stable under reverse insertion', async () => {
    const profiles = ['worker-000729', 'worker-004592']
    const source = roster(profiles)

    const forward = await enrichBotRosterEntities(source, entities(source), async () => ({
      profiles: profiles.map(name => ({ name }))
    }))

    const reversedRoster = roster([...profiles].reverse())

    const reverse = await enrichBotRosterEntities(reversedRoster, entities(reversedRoster), async () => ({
      profiles: [...profiles].reverse().map(name => ({ name }))
    }))

    const forwardByKey = new Map(forward.map(entity => [entity.key, entity]))

    expect(new Set(forward.map(entity => JSON.stringify(entity.position))).size).toBe(2)
    expect(
      forward.every(entity => {
        const slot = entity.presentation?.placement.slot

        return slot !== undefined && slot >= 0 && slot < 512 * 512
      })
    ).toBe(true)
    expect(
      reverse.every(
        entity => JSON.stringify(entity.position) === JSON.stringify(forwardByKey.get(entity.key)?.position)
      )
    ).toBe(true)
  })

  it('retains an existing colliding identity slot when an earlier exact key is added incrementally', async () => {
    const placementState = createBotRosterPlacementState()
    const firstRoster = roster(['worker-004592'])

    const first = await enrichBotRosterEntities(
      firstRoster,
      entities(firstRoster),
      async () => ({ profiles: [] }),
      new Map(),
      { placementState }
    )

    const expandedRoster = roster(['worker-000729', 'worker-004592'])

    const expanded = await enrichBotRosterEntities(
      expandedRoster,
      entities(expandedRoster),
      async () => ({ profiles: [] }),
      new Map(),
      { placementState }
    )

    const retained = expanded.find(entity => entity.identity.profile === 'worker-004592')
    const inserted = expanded.find(entity => entity.identity.profile === 'worker-000729')

    expect(first[0]?.presentation?.placement.slot).toBe(39024)
    expect(retained?.presentation?.placement.slot).toBe(39024)
    expect(retained?.position).toEqual(first[0]?.position)
    expect(inserted?.presentation?.placement.slot).not.toBe(39024)
  })

  it('retains slots across reorder and removal/re-add, then cold-starts deterministically after reconnect', async () => {
    const placementState = createBotRosterPlacementState()
    const profiles = ['worker-000729', 'worker-004592']
    const initialRoster = roster(profiles)
    const read = async () => ({ profiles: [] })

    const initial = await enrichBotRosterEntities(initialRoster, entities(initialRoster), read, new Map(), {
      placementState
    })

    const reversedRoster = roster([...profiles].reverse())

    const reversed = await enrichBotRosterEntities(reversedRoster, entities(reversedRoster), read, new Map(), {
      placementState
    })

    const retainedRoster = roster(['worker-004592'])

    const retained = await enrichBotRosterEntities(retainedRoster, entities(retainedRoster), read, new Map(), {
      placementState
    })

    const readded = await enrichBotRosterEntities(initialRoster, entities(initialRoster), read, new Map(), {
      placementState
    })

    const emptyRoster = roster([])
    await enrichBotRosterEntities(emptyRoster, entities(emptyRoster), read, new Map(), { placementState })

    const reconnected = await enrichBotRosterEntities(reversedRoster, entities(reversedRoster), read, new Map(), {
      placementState
    })

    const slots = (rows: readonly LunarEntity[]) =>
      new Map(rows.map(entity => [entity.key, entity.presentation?.placement.slot]))

    expect(slots(reversed)).toEqual(slots(initial))
    expect(retained[0]?.presentation?.placement.slot).toBe(
      initial.find(entity => entity.identity.profile === 'worker-004592')?.presentation?.placement.slot
    )
    expect(slots(readded)).toEqual(slots(initial))
    expect(slots(reconnected)).toEqual(slots(initial))
  })

  it('keeps the generated negative-step cohort physically unique', async () => {
    const profiles = Array.from({ length: 8192 }, (_, index) => `worker-${index.toString().padStart(6, '0')}`)
    const source = roster(profiles)
    const result = await enrichBotRosterEntities(source, entities(source), async () => ({ profiles: [] }))

    expect(new Set(result.map(entity => entity.presentation?.placement.slot)).size).toBe(8192)
    expect(new Set(result.map(entity => JSON.stringify(entity.position))).size).toBe(8192)
    expect(
      result.every(entity => {
        const slot = entity.presentation?.placement.slot

        return slot !== undefined && slot >= 0 && slot < 512 * 512
      })
    ).toBe(true)
  })

  it('uses every slot exactly once under a large generated collision load', () => {
    const keys = Array.from(
      { length: 3000 },
      (_, index) => `profile:connection=collision:profile=worker-${index}` as LunarEntity['key']
    )

    const slots = assignStablePlacementSlots(keys, 4096)
    const assigned = [...slots.values()].filter((slot): slot is number => slot !== undefined)

    expect(assigned).toHaveLength(3000)
    expect(new Set(assigned).size).toBe(3000)
    expect(assigned.every(slot => slot >= 0 && slot < 4096)).toBe(true)
  })

  it('fails boundedly into aggregate-only placement when a test lattice is exhausted', async () => {
    const profiles = ['alpha', 'beta', 'gamma', 'delta', 'epsilon']
    const source = roster(profiles)

    const result = await enrichBotRosterEntities(
      source,
      entities(source),
      async () => ({ profiles: profiles.map(name => ({ name })) }),
      new Map(),
      { latticeSide: 2 }
    )

    const physicallyPlaced = result.filter(entity => entity.position !== undefined)
    const aggregateOnly = result.filter(entity => entity.position === undefined)
    const reversedRoster = roster([...profiles].reverse())

    const reversed = await enrichBotRosterEntities(
      reversedRoster,
      entities(reversedRoster),
      async () => ({ profiles: [...profiles].reverse().map(name => ({ name })) }),
      new Map(),
      { latticeSide: 2 }
    )

    const reversedAggregateOnly = reversed.filter(entity => entity.position === undefined)

    expect(physicallyPlaced).toHaveLength(4)
    expect(new Set(physicallyPlaced.map(entity => entity.presentation?.placement.slot)).size).toBe(4)
    expect(aggregateOnly).toHaveLength(1)
    expect(aggregateOnly[0]?.presentation?.placement).toMatchObject({ lodHint: 1, overflow: true })
    expect(aggregateOnly[0]?.presentation?.placement.slot).toBeUndefined()
    expect(reversedAggregateOnly.map(entity => entity.key)).toEqual(aggregateOnly.map(entity => entity.key))
    expect(result).toHaveLength(5)
  })

  it('keeps retained slots stable while a released slot promotes one aggregate-only identity', async () => {
    const placementState = createBotRosterPlacementState()
    const profiles = ['alpha', 'beta', 'gamma', 'delta', 'epsilon']

    const enrich = async (active: readonly string[]) => {
      const source = roster(active)

      return enrichBotRosterEntities(source, entities(source), async () => ({ profiles: [] }), new Map(), {
        latticeSide: 2,
        placementState
      })
    }

    const full = await enrich(profiles)
    const aggregate = full.find(entity => entity.position === undefined)!
    const removed = full.find(entity => entity.position !== undefined)!

    const retainedSlots = new Map(
      full
        .filter(entity => entity.key !== removed.key && entity.position !== undefined)
        .map(entity => [entity.key, entity.position])
    )

    const afterRemoval = await enrich(profiles.filter(profile => profile !== removed.identity.profile))

    expect(afterRemoval.find(entity => entity.key === aggregate.key)?.position).toBeDefined()
    expect(
      afterRemoval
        .filter(entity => retainedSlots.has(entity.key))
        .every(entity => JSON.stringify(entity.position) === JSON.stringify(retainedSlots.get(entity.key)))
    ).toBe(true)

    const afterReadd = await enrich(profiles)
    expect(afterReadd.find(entity => entity.key === removed.key)?.position).toBeUndefined()
  })

  it('does not bind unscoped projected members to another exact connection', async () => {
    const source = roster(['steward'], 'desktop-source')

    const [entity] = await enrichBotRosterEntities(source, entities(source), async () => ({
      profiles: [
        {
          name: 'steward',
          ui_meta: {
            'hermes-bots-groups': {
              rooms: {
                'id:local-only': {
                  members: [{ connectionId: 'local', name: 'steward' }],
                  name: 'Research Lab',
                  roomId: 'local-only'
                }
              },
              version: 3
            }
          }
        }
      ]
    }))

    expect(entity?.presentation?.groups).toEqual([])
    expect(entity?.destination).toBe('unavailable')
  })
})
