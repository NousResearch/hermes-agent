import { describe, expect, it, vi } from 'vitest'

import actualManifest from '../../../../public/lunar-city/v2/world-manifest.v2.json'
import { parseWorldManifest } from '../manifest'
import type { EntityKey, LunarCitySnapshot, LunarEntity, Vec3 } from '../model'

import { applyLodSelection, createEntityRegistry, type EntityPositionRequest, type EntityPresentationFactory, selectLodIndex } from './entities'
import { createEntityPositionState } from './entity-position-state'

function entity(index: number, overrides: Partial<LunarEntity> = {}): LunarEntity {
  const key = `session:connection=${index}:worker` as EntityKey

  return {
    animation: 'idle',
    authority: 'authoritative',
    destination: 'garden',
    identity: {
      kind: 'session',
      connectionId: `connection-${index}`,
      profile: 'worker',
      sessionId: `session-${index}`
    },
    key,
    observedAt: 1,
    position: { x: index, y: 0, z: 0 },
    ...overrides
  }
}

function snapshot(...entities: LunarEntity[]): LunarCitySnapshot {
  return {
    entities: new Map(entities.map(value => [value.key, value])),
    observedAt: 1,
    revision: 1,
    sources: []
  }
}

function factory(): EntityPresentationFactory & {
  animated: ReturnType<typeof vi.fn>
  groups: Map<string, { dispose: ReturnType<typeof vi.fn>; sync: ReturnType<typeof vi.fn> }>
} {
  const groups = new Map<string, { dispose: ReturnType<typeof vi.fn>; sync: ReturnType<typeof vi.fn> }>()

  const animated = vi.fn(() => ({
    dispose: vi.fn(),
    setAnimation: vi.fn(),
    setLod: vi.fn(),
    setPosition: vi.fn(),
    setStaticPose: vi.fn()
  }))

  return {
    animated,
    createAnimated: animated,
    createInstancedGroup: vi.fn((key: string) => {
      const group = { dispose: vi.fn(), sync: vi.fn() }
      groups.set(key, group)

      return group
    }),
    groups
  }
}

describe('EntityRegistry', () => {
  it('separates the 24-visual budget from continuous worker animation activity', () => {
    const presentationFactory = factory()
    const assets = parseWorldManifest(structuredClone(actualManifest)).characterAssets

    const registry = createEntityRegistry({
      characterAssets: assets,
      factory: presentationFactory,
      workerClips: new Set(['idle', 'walk', 'work'])
    })

    const resting = Array.from({ length: 90 }, (_, index) =>
      entity(index, {
        animation: 'idle',
        identity: { kind: 'profile', connectionId: 'local', profile: `resting-${index}` },
        presentation: {
          groups: [{ id: 'engineering', name: 'Engineering Guild' }],
          metadata: { source: 'profiles:local', state: 'fresh' },
          placement: { lodHint: 0, overflow: false, primaryGroupId: 'engineering', slot: index }
        }
      })
    )

    registry.reconcile(snapshot(...resting))

    expect(presentationFactory.animated).toHaveBeenCalledTimes(24)
    expect(registry.hasActiveAnimations()).toBe(false)

    registry.reconcile(snapshot({ ...resting[0]!, animation: 'work' }, ...resting.slice(1)))
    expect(registry.hasActiveAnimations()).toBe(true)

    registry.reconcile(snapshot(...resting))
    expect(registry.hasActiveAnimations()).toBe(false)
  })

  it('retains beyond-capacity exact profiles only in truthful aggregate counts', () => {
    const presentationFactory = factory()
    const assets = parseWorldManifest(structuredClone(actualManifest)).characterAssets

    const registry = createEntityRegistry({
      characterAssets: assets,
      factory: presentationFactory,
      workerClips: new Set(['idle', 'walk'])
    })

    const profiles = Array.from({ length: 140 }, (_, index) =>
      entity(index, {
        identity: { kind: 'profile', connectionId: 'local', profile: `worker-${index}` },
        position: index < 128 ? { x: index, y: 0, z: 0 } : undefined,
        presentation: {
          groups: [{ id: 'engineering', name: 'Engineering Guild' }],
          metadata: { source: 'profiles:local', state: 'fresh' },
          placement: {
            lodHint: index < 128 ? 1 : 2,
            overflow: index >= 128,
            primaryGroupId: 'engineering',
            ...(index < 128 ? { slot: index } : {})
          }
        }
      })
    )

    registry.reconcile(snapshot(...profiles))

    expect(registry.aggregate('garden')).toEqual({ animations: { idle: 140 }, total: 140 })
    expect(registry.entity(profiles[139]!.key)).toBeDefined()
    expect(registry.entity(profiles[139]!.key)?.visual).toBeUndefined()
    expect(registry.navigationEntity(profiles[139]!.key)).toBeUndefined()
  })

  it('does not sort LOD thresholds during repeated frame selection', () => {
    const entries = Object.freeze([{ distance: 0 }, { distance: 28 }, { distance: 64 }])
    const sort = vi.spyOn(Array.prototype, 'sort')

    for (let index = 0; index < 100; index += 1) {
      expect(selectLodIndex(entries, { distance: 40, lodAdvance: 0, selected: false })).toBe(1)
    }

    expect(sort).not.toHaveBeenCalled()
    sort.mockRestore()
  })
  it('passes exact group-kit identity data through shared batches without allocating a group per profile', () => {
    const presentationFactory = factory()
    const assets = parseWorldManifest(structuredClone(actualManifest)).characterAssets

    const workers = Array.from({ length: 90 }, (_, index) =>
      entity(index, {
        identity: { kind: 'profile', connectionId: 'local', profile: `worker-${index}` },
        presentation: {
          groups: [{ id: 'engineering', name: 'Engineering Guild' }],
          metadata: { source: 'profiles:local', state: 'fresh' },
          placement: { lodHint: 1, overflow: true, primaryGroupId: 'engineering', slot: index }
        }
      })
    )

    const registry = createEntityRegistry({
      characterAssets: assets,
      factory: presentationFactory,
      workerClips: new Set(['idle', 'walk'])
    })

    registry.reconcile(snapshot(...workers))

    expect(presentationFactory.groups).toHaveLength(1)
    const members = [...presentationFactory.groups.values()][0]!.sync.mock.calls.at(-1)?.[0]
    expect(members).toHaveLength(90)
    expect(new Set(members.map((member: any) => member.character.visibleSignature))).toHaveLength(90)
    expect(members.every((member: any) => member.character.kitId === 'engineering-guild')).toBe(true)
  })

  it('renders only the global 24 near exact profiles with full animated signatures and leaves mid profiles instanced', () => {
    const presentationFactory = factory()
    const assets = parseWorldManifest(structuredClone(actualManifest)).characterAssets

    const workers = Array.from({ length: 32 }, (_, index) =>
      entity(index, {
        identity: { kind: 'profile', connectionId: 'local', profile: `worker-${index}` },
        presentation: {
          groups: [{ id: 'engineering', name: 'Engineering Guild' }],
          metadata: { source: 'profiles:local', state: 'fresh' },
          placement: {
            lodHint: index < 24 ? 0 : 1,
            overflow: index >= 24,
            primaryGroupId: 'engineering',
            slot: index
          }
        }
      })
    )

    const registry = createEntityRegistry({
      characterAssets: assets,
      factory: presentationFactory,
      workerClips: new Set(['idle', 'walk'])
    })

    registry.reconcile(snapshot(...workers))

    expect(presentationFactory.animated).toHaveBeenCalledTimes(24)
    expect(registry.instancedGroup('worker:idle:kit:engineering-guild:lod:1')?.count).toBe(8)
  })
  it('retains keyed objects and a single hardware-instanced idle group across identical snapshots', () => {
    const presentationFactory = factory()
    const registry = createEntityRegistry({ factory: presentationFactory, workerClips: new Set(['idle', 'walk']) })
    const workers = Array.from({ length: 100 }, (_, index) => entity(index))

    registry.reconcile(snapshot(...workers))
    const groupsAfterFirst = presentationFactory.groups.size
    registry.reconcile(snapshot(...workers))

    expect(registry.instancedGroup('worker:idle')?.count).toBe(100)
    expect(presentationFactory.groups.size).toBe(groupsAfterFirst)
    expect(presentationFactory.groups.get('worker:idle')?.sync).toHaveBeenLastCalledWith(
      expect.arrayContaining(workers.map(worker => expect.objectContaining({ key: worker.key })))
    )
    expect(presentationFactory.animated).not.toHaveBeenCalled()
  })

  it('keeps selected and moving workers individually animated while the rest remain instanced', () => {
    const presentationFactory = factory()
    const selected = entity(1)
    const moving = entity(2, { animation: 'walk' })
    const idle = entity(3)
    const registry = createEntityRegistry({ factory: presentationFactory, workerClips: new Set(['idle', 'walk']) })

    registry.setSelection(selected.key)
    registry.reconcile(snapshot(selected, moving, idle))

    expect(presentationFactory.animated).toHaveBeenCalledTimes(2)
    expect(registry.instancedGroup('worker:idle')?.count).toBe(1)
  })

  it('moves non-selected idle groups to their next declared LOD without dropping their exact members', () => {
    const presentationFactory = factory()
    const registry = createEntityRegistry({ factory: presentationFactory, workerClips: new Set(['idle', 'walk']) })
    const workers = [entity(1), entity(2)]

    registry.reconcile(snapshot(...workers))
    registry.applyLodPolicy(() => 1)

    expect(registry.instancedGroup('worker:idle')).toBeUndefined()
    expect(registry.instancedGroup('worker:idle:lod:1')?.count).toBe(2)
    expect(presentationFactory.groups.get('worker:idle')?.dispose).toHaveBeenCalledOnce()
  })

  it('starts bounded district overflow at its declared aggregate LOD without per-frame promotion work', () => {
    const presentationFactory = factory()

    const overflow = entity(1, {
      presentation: {
        groups: [{ id: 'engineering', name: 'Engineering Guild' }],
        metadata: { source: 'profiles:connection-1', state: 'fresh' },
        placement: { lodHint: 1, overflow: true, primaryGroupId: 'engineering', slot: 24 }
      }
    })

    const registry = createEntityRegistry({ factory: presentationFactory, workerClips: new Set(['idle', 'walk']) })

    registry.reconcile(snapshot(overflow))

    expect(registry.instancedGroup('worker:idle:lod:1')?.count).toBe(1)
    expect(presentationFactory.animated).not.toHaveBeenCalled()
  })

  it('counts a capacity-exhausted profile without inventing an origin render or navigation slot', () => {
    const presentationFactory = factory()

    const aggregateOnly = entity(1, {
      identity: { kind: 'profile', connectionId: 'local', profile: 'aggregate-only' },
      position: undefined,
      presentation: {
        groups: [],
        metadata: { source: 'profiles:local', state: 'fresh' },
        placement: { lodHint: 1, overflow: true }
      }
    })

    const registry = createEntityRegistry({ factory: presentationFactory, workerClips: new Set(['idle', 'walk']) })
    const resolveIndex = vi.fn(() => 0)
    const isNearby = vi.fn(() => true)

    registry.reconcile(snapshot(aggregateOnly))
    registry.setSelection(aggregateOnly.key)
    registry.applyLodPolicy(resolveIndex, isNearby)

    expect(registry.aggregate('garden')).toEqual({ animations: { idle: 1 }, total: 1 })
    expect(registry.instancedGroup('worker:idle:lod:1')).toBeUndefined()
    expect(registry.navigationEntity(aggregateOnly.key)).toBeUndefined()
    expect(registry.hasActiveAnimations()).toBe(false)
    expect(resolveIndex).not.toHaveBeenCalled()
    expect(isNearby).not.toHaveBeenCalled()
    expect(presentationFactory.animated).not.toHaveBeenCalled()
  })

  it('does not let selected aggregate-only rows displace a worker from the global 24-visual budget', () => {
    const presentationFactory = factory()
    const workers = Array.from({ length: 24 }, (_, index) => entity(index, { animation: 'walk' }))

    const aggregateOnly = entity(100, {
      identity: { kind: 'profile', connectionId: 'local', profile: 'aggregate-only' },
      position: undefined,
      presentation: {
        groups: [],
        metadata: { source: 'profiles:local', state: 'fresh' },
        placement: { lodHint: 1, overflow: true }
      }
    })

    const registry = createEntityRegistry({ factory: presentationFactory, workerClips: new Set(['idle', 'walk']) })

    registry.reconcile(snapshot(...workers, aggregateOnly))
    registry.setSelection(aggregateOnly.key)

    const activeVisuals = presentationFactory.animated.mock.results
      .map(result => result.value)
      .filter(visual => !visual.dispose.mock.calls.length)

    expect(activeVisuals).toHaveLength(24)
    expect(registry.entity(aggregateOnly.key)?.visual).toBeUndefined()
  })

  it('moves a retained worker to aggregate LOD when a later roster makes its stable district rank overflow', () => {
    const presentationFactory = factory()
    const registry = createEntityRegistry({ factory: presentationFactory, workerClips: new Set(['idle', 'walk']) })
    const near = entity(1)

    const overflow = entity(1, {
      presentation: {
        groups: [{ id: 'engineering', name: 'Engineering Guild' }],
        metadata: { source: 'profiles:connection-1', state: 'fresh' },
        placement: { lodHint: 1, overflow: true, primaryGroupId: 'engineering', slot: 24 }
      }
    })

    registry.reconcile(snapshot(near))
    registry.reconcile(snapshot(overflow))

    expect(registry.instancedGroup('worker:idle:lod:1')?.count).toBe(1)
  })

  it('promotes one selected overflow worker to LOD0 by replacing a near worker within the strict district budget', () => {
    const presentationFactory = factory()

    const workers = Array.from({ length: 26 }, (_, index) =>
      entity(index, {
        animation: index === 24 ? 'walk' : 'idle',
        presentation: {
          groups: [{ id: 'engineering', name: 'Engineering Guild' }],
          metadata: { source: 'profiles:local', state: 'fresh' },
          placement: { lodHint: index >= 24 ? 1 : 0, overflow: index >= 24, slot: index }
        }
      })
    )

    const selectedOverflow = workers[25]!
    const otherOverflow = workers[24]!
    const registry = createEntityRegistry({ factory: presentationFactory, workerClips: new Set(['idle', 'walk']) })

    registry.reconcile(snapshot(...workers))
    registry.applyLodPolicy(
      () => 0,
      () => true
    )
    registry.setSelection(selectedOverflow.key)

    const activeVisuals = presentationFactory.animated.mock.results
      .map(result => result.value)
      .filter(visual => !visual.dispose.mock.calls.length)

    expect(activeVisuals).toHaveLength(24)
    expect(registry.entity(selectedOverflow.key)?.visual?.setLod).toHaveBeenLastCalledWith(0)
    expect(registry.entity(otherOverflow.key)?.visual).toBeUndefined()
    expect(registry.instancedGroup('worker:idle:lod:1')?.count).toBe(1)
    expect(registry.instancedGroup('worker:walk:lod:1')?.count).toBe(1)
  })

  it('caps mixed profile, session, subagent, and Kanban workers at 24 animated visuals globally across districts', () => {
    const presentationFactory = factory()

    const workers = Array.from({ length: 32 }, (_, index) => {
      const base = entity(index, { animation: 'walk', destination: index % 2 ? 'lab' : 'project' })

      if (index < 8) {
        return {
          ...base,
          identity: { kind: 'profile' as const, connectionId: `profile-${index}`, profile: 'worker' }
        }
      }

      if (index < 16) {
        return base
      }

      if (index < 24) {
        return {
          ...base,
          identity: {
            kind: 'subagent' as const,
            connectionId: `connection-${index}`,
            profile: 'worker',
            sessionId: `session-${index}`,
            subagentId: `subagent-${index}`
          }
        }
      }

      return {
        ...base,
        identity: {
          kind: 'kanban' as const,
          board: 'delivery',
          connectionId: `connection-${index}`,
          profile: 'worker',
          taskId: `task-${index}`
        }
      }
    })

    const selected = workers[31]!
    const registry = createEntityRegistry({ factory: presentationFactory, workerClips: new Set(['idle', 'walk']) })

    registry.reconcile(snapshot(...workers))
    expect(
      presentationFactory.animated.mock.results.filter(result => !result.value.dispose.mock.calls.length)
    ).toHaveLength(24)
    expect(registry.entity(selected.key)?.visual).toBeUndefined()

    registry.setSelection(selected.key)

    const activeVisuals = presentationFactory.animated.mock.results
      .map(result => result.value)
      .filter(visual => !visual.dispose.mock.calls.length)

    expect(activeVisuals).toHaveLength(24)
    expect(registry.entity(selected.key)?.visual).toBeDefined()
  })

  it('does not rerank or rescan the population for constant-time status queries and per-frame LOD updates', () => {
    const presentationFactory = factory()
    const workers = Array.from({ length: 100 }, (_, index) => entity(index, { animation: 'walk' }))
    const sort = vi.spyOn(Array.prototype, 'sort')
    const registry = createEntityRegistry({ factory: presentationFactory, workerClips: new Set(['idle', 'walk']) })

    registry.reconcile(snapshot(...workers))
    sort.mockClear()
    registry.applyLodPolicy(
      () => 0,
      () => true
    )
    registry.hasActiveAnimations()
    registry.instancedGroup('worker:walk')
    registry.instancedGroup('worker:walk:lod:1')

    expect(sort).not.toHaveBeenCalled()
    sort.mockRestore()
  })

  it('reports the exact live count of retained active worker animations', () => {
    const registry = createEntityRegistry({
      characterAssets: parseWorldManifest(structuredClone(actualManifest)).characterAssets,
      factory: factory(),
      workerClips: new Set(['idle', 'walk', 'work'])
    })

    const placed = (index: number, animation: string) =>
      entity(index, {
        animation,
        identity: { connectionId: 'local', kind: 'profile', profile: `worker-${index}` },
        presentation: {
          groups: [{ id: 'engineering', name: 'Engineering Guild' }],
          metadata: { source: 'test', state: 'fresh' },
          placement: { lodHint: 0, overflow: false, primaryGroupId: 'engineering', slot: index }
        }
      })

    const idle = placed(1, 'idle')
    const working = placed(2, 'work')
    const moving = placed(3, 'walk')

    registry.reconcile(snapshot(idle, working, moving))

    expect(registry.activeAnimationCount()).toBe(2)
    registry.reconcile(snapshot(idle, working, { ...moving, animation: 'idle' }))
    registry.setMoving(moving.key, false)
    expect(registry.activeAnimationCount()).toBe(1)
    registry.dispose()
    expect(registry.activeAnimationCount()).toBe(0)
  })

  it('releases removed presentation resources once without disturbing unrelated entities', () => {
    const presentationFactory = factory()
    const first = entity(1, { animation: 'walk' })
    const second = entity(2, { animation: 'walk' })
    const registry = createEntityRegistry({ factory: presentationFactory, workerClips: new Set(['idle', 'walk']) })

    registry.reconcile(snapshot(first, second))
    const firstVisual = presentationFactory.animated.mock.results[0]?.value
    const secondVisual = presentationFactory.animated.mock.results[1]?.value
    registry.reconcile(snapshot(second))
    registry.reconcile(snapshot(second))

    expect(firstVisual.dispose).toHaveBeenCalledOnce()
    expect(secondVisual.dispose).not.toHaveBeenCalled()
  })

  it('keeps exact observed counts and state distributions when its population is represented at aggregate detail', () => {
    const presentationFactory = factory()
    const registry = createEntityRegistry({ factory: presentationFactory, workerClips: new Set(['idle', 'walk']) })
    const workers = Array.from({ length: 250 }, (_, index) => entity(index, { animation: index % 2 ? 'idle' : 'walk' }))

    registry.reconcile(snapshot(...workers))

    expect(registry.aggregate('garden')).toEqual({ animations: { idle: 125, walk: 125 }, total: 250 })
  })

  it('uses a declared reduced pose and one bounded diagnostic when an entity asks for a missing GLB clip', () => {
    const presentationFactory = factory()
    const diagnostic = vi.fn()
    const worker = entity(1, { animation: 'made-up' })

    const registry = createEntityRegistry({
      diagnostic,
      factory: presentationFactory,
      workerClips: new Set(['idle'])
    })

    registry.setSelection(worker.key)
    registry.reconcile(snapshot(worker))
    registry.reconcile(snapshot(worker))

    expect(presentationFactory.animated.mock.results[0]?.value.setStaticPose).toHaveBeenCalledWith('idle')
    expect(diagnostic).toHaveBeenCalledOnce()
  })

  it('publishes a live focus anchor for the retained exact key and removes it before a same-named entity can be selected', () => {
    const focusAnchors = new Map<EntityKey, () => Vec3 | undefined>()
    const presentationFactory = factory()
    const first = entity(1, { animation: 'walk', position: { x: 1, y: 0, z: 0 } })

    const replacement = entity(2, {
      animation: 'walk',
      identity: { kind: 'session', connectionId: 'connection-2', profile: 'worker', sessionId: 'session-2' }
    })

    const registry = createEntityRegistry({
      focusAnchors,
      factory: presentationFactory,
      workerClips: new Set(['idle', 'walk'])
    })

    registry.reconcile(snapshot(first))
    registry.reconcile(snapshot({ ...first, position: { x: 8, y: 0, z: 0 } }))
    expect(focusAnchors.get(first.key)?.()).toEqual({ x: 8, y: 0, z: 0 })

    registry.reconcile(snapshot(replacement))
    expect(focusAnchors.get(first.key)).toBeUndefined()
    expect(focusAnchors.get(replacement.key)?.()).toEqual({ x: 2, y: 0, z: 0 })
  })

  it('publishes authoritative typed worker focus metadata that follows motion and fails closed for stale workers', () => {
    const focusMetadata = new Map<
      EntityKey,
      () => { cameraAnchor: Vec3; focusEntityKey: EntityKey; occlusionGroup: string } | undefined
    >()

    const presentationFactory = factory()
    const worker = entity(1, { position: { x: 2, y: 0, z: 3 } })

    const registry = createEntityRegistry({
      focusMetadata,
      factory: presentationFactory,
      workerClips: new Set(['idle', 'walk'])
    })

    registry.reconcile(snapshot(worker))
    expect(focusMetadata.get(worker.key)?.()).toEqual({
      cameraAnchor: { x: 2, y: 0, z: 3 },
      focusEntityKey: worker.key,
      occlusionGroup: 'workers'
    })

    registry.setPosition(worker.key, { x: 9, y: 0, z: -2 })
    expect(focusMetadata.get(worker.key)?.()?.cameraAnchor).toEqual({ x: 9, y: 0, z: -2 })

    registry.reconcile(snapshot({ ...worker, authority: 'stale' }))
    expect(focusMetadata.get(worker.key)?.()).toBeUndefined()
  })

  it('exposes only retained presentation state to navigation and flushes its changed position without touching snapshots', () => {
    const presentationFactory = factory()
    const worker = entity(1, { animation: 'walk' })
    const registry = createEntityRegistry({ factory: presentationFactory, workerClips: new Set(['idle', 'walk']) })

    registry.reconcile(snapshot(worker))
    const navigationEntity = registry.navigationEntity(worker.key)!
    navigationEntity.position.x = 12
    registry.syncMotion()

    expect(worker.position).toEqual({ x: 1, y: 0, z: 0 })
    expect(presentationFactory.animated.mock.results[0]?.value.setPosition).toHaveBeenLastCalledWith({
      x: 12,
      y: 0,
      z: 0
    })
  })

  it('keeps the authoritative arrival clip while navigation temporarily presents the declared walk clip', () => {
    const presentationFactory = factory()
    const worker = entity(1, { animation: 'work' })

    const registry = createEntityRegistry({
      factory: presentationFactory,
      workerClips: new Set(['idle', 'walk', 'work'])
    })

    registry.setSelection(worker.key)
    registry.reconcile(snapshot(worker))
    const navigationEntity = registry.navigationEntity(worker.key)!
    navigationEntity.animation = 'walk'
    registry.reconcile(snapshot({ ...worker, animation: 'work' }))
    registry.syncMotion()

    expect(presentationFactory.animated.mock.results[0]?.value.setAnimation).toHaveBeenLastCalledWith('walk')

    navigationEntity.animation = 'work'
    registry.setMoving(worker.key, false)
    registry.syncMotion()

    expect(presentationFactory.animated.mock.results[0]?.value.setAnimation).toHaveBeenLastCalledWith('work')
  })

  it('uses the supplied quality animation-distance policy to promote nearby workers and demote distant workers', () => {
    const presentationFactory = factory()
    const worker = entity(1, { animation: 'idle' })
    const registry = createEntityRegistry({ factory: presentationFactory, workerClips: new Set(['idle', 'walk']) })

    registry.reconcile(snapshot(worker))
    registry.applyLodPolicy(
      () => 0,
      () => true
    )
    expect(presentationFactory.animated).toHaveBeenCalledOnce()

    const visual = presentationFactory.animated.mock.results[0]?.value
    registry.applyLodPolicy(
      () => 1,
      () => false
    )

    expect(visual.dispose).toHaveBeenCalledOnce()
    expect(registry.instancedGroup('worker:idle:lod:1')?.count).toBe(1)
  })
})

describe('LOD selection', () => {
  function fakeLod() {
    const near = { setEnabled: vi.fn() }
    const far = { setEnabled: vi.fn() }

    return { far, near }
  }

  it.each([
    ['near camera', 1, false, 0, [true, false]],
    ['far camera', 99, false, 0, [false, true]],
    ['selected entity', 99, true, 1, [true, false]],
    ['degraded entity', 1, false, 1, [false, true]]
  ])('enables exactly one near/far LOD for %s', (_label, distance, selected, lodAdvance, expected) => {
    const { far, near } = fakeLod()

    applyLodSelection(
      [
        { distance: 0, node: near },
        { distance: 28, node: far }
      ],
      { distance, lodAdvance, selected }
    )

    expect(near.setEnabled).toHaveBeenLastCalledWith(expected[0])
    expect(far.setEnabled).toHaveBeenLastCalledWith(expected[1])
  })
})

it('preserves actual travel and resolved arrival positions across unchanged source polling', () => {
  const registry = createEntityRegistry({ factory: factory(), workerClips: new Set(['idle', 'walk', 'work']) })
  const source = entity(1)
  registry.reconcile(snapshot(source))
  registry.setMoving(source.key, true)
  registry.setPosition(source.key, { x: 5, y: .25, z: 4 })
  registry.reconcile(snapshot({ ...source, observedAt: 2, position: { ...source.position! } }))
  expect(registry.entity(source.key)?.position).toEqual({ x: 5, y: .25, z: 4 })
  registry.setMoving(source.key, false)
  registry.setPosition(source.key, { x: 9, y: .5, z: 8 })
  registry.reconcile(snapshot({ ...source, observedAt: 3 }))
  expect(registry.entity(source.key)?.position).toEqual({ x: 9, y: .5, z: 8 })
  expect(source.position).toEqual({ x: 1, y: 0, z: 0 })
})

it('applies changed authoritative origins once while stale positions cannot overwrite navigation', () => {
  const registry = createEntityRegistry({ factory: factory(), workerClips: new Set(['idle', 'walk']) })
  const source = entity(1)
  registry.reconcile(snapshot(source))
  registry.setPosition(source.key, { x: 5, y: 0, z: 4 })
  const corrected = { ...source, position: { x: 12, y: 0, z: 0 } }
  registry.reconcile(snapshot({ ...corrected, authority: 'stale' }))
  expect(registry.entity(source.key)?.position).toEqual({ x: 5, y: 0, z: 4 })
  registry.reconcile(snapshot(corrected))
  expect(registry.entity(source.key)?.position).toEqual(corrected.position)
  registry.setPosition(source.key, { x: 13, y: 0, z: 0 })
  registry.reconcile(snapshot({ ...corrected, position: { ...corrected.position } }))
  expect(registry.entity(source.key)?.position).toEqual({ x: 13, y: 0, z: 0 })
})

it('restores resolved navigation coordinates on remount while source corrections pass through validation once', () => {
  const state = createEntityPositionState({ scope: 'registry-remount' })
  const resolve = vi.fn(({ candidate, previous }: EntityPositionRequest) => candidate.x > 10 ? undefined : previous ?? candidate)
  const options = { factory: factory(), workerClips: new Set(['idle']), positionState: state, resolvePosition: resolve }
  const worker = entity(1)
  const first = createEntityRegistry(options)
  first.reconcile(snapshot(worker))
  first.navigationEntity(worker.key)!.position.x = 7
  first.syncMotion()
  first.dispose()
  const second = createEntityRegistry(options)
  second.reconcile(snapshot(worker))
  expect(second.entity(worker.key)?.position.x).toBe(7)
  expect(resolve).toHaveBeenLastCalledWith(expect.objectContaining({ reason: 'restore', candidate: { x: 7, y: 0, z: 0 } }))
  const correction = { ...worker, position: { x: 20, y: 0, z: 0 } }
  second.reconcile(snapshot(correction))
  expect(second.entity(worker.key)?.position.x).toBe(7)
  resolve.mockClear()
  second.reconcile(snapshot(correction))
  expect(resolve).not.toHaveBeenCalled()
  second.reconcile(snapshot())
  second.dispose()
  expect(state.read(worker.key)).toBeUndefined()
})

it('does not spawn rejected coordinates or prolong cached positions with stale observations', () => {
  let time = 0
  const state = createEntityPositionState({ scope: 'registry-stale', ttlMs: 10, now: () => time })
  const resolve = vi.fn(({ candidate }: EntityPositionRequest) => candidate.x < 10 ? candidate : undefined)
  const options = { factory: factory(), workerClips: new Set(['idle']), positionState: state, resolvePosition: resolve }
  const registry = createEntityRegistry(options)
  const worker = entity(1)
  registry.reconcile(snapshot({ ...worker, position: { x: 99, y: 0, z: 0 } }))
  expect(registry.entity(worker.key)).toBeUndefined()
  expect(registry.navigationEntity(worker.key)).toBeUndefined()
  registry.reconcile(snapshot(worker))
  time = 9
  registry.reconcile(snapshot({ ...worker, authority: 'stale' }))
  registry.syncMotion()
  registry.dispose()
  time = 11
  expect(state.read(worker.key)).toBeUndefined()
  expect(worker.position).toEqual({ x: 1, y: 0, z: 0 })
})
