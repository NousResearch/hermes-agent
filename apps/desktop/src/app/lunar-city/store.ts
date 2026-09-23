import { atom } from 'nanostores'

import type { EntityKey, LunarCitySnapshot, LunarEntity, SourceHealth, Vec3 } from './model'

export interface LunarDelta {
  observedAt: number
  removals: readonly EntityKey[]
  revision: number
  sources: readonly SourceHealth[]
  upserts: readonly LunarEntity[]
}

interface SnapshotSeed {
  observedAt?: number
  revision?: number
}

function freezeVec3(value: Vec3): Vec3 {
  return Object.freeze({ x: value.x, y: value.y, z: value.z })
}

function freezeEntity(entity: LunarEntity): LunarEntity {
  const identity = Object.freeze({ ...entity.identity }) as LunarEntity['identity']

  const presentation = entity.presentation
    ? Object.freeze({
        ...entity.presentation,
        groups: Object.freeze(entity.presentation.groups.map(group => Object.freeze({ ...group }))),
        metadata: Object.freeze({ ...entity.presentation.metadata }),
        placement: Object.freeze({ ...entity.presentation.placement })
      })
    : undefined

  return Object.freeze({
    ...entity,
    identity,
    ...(presentation ? { presentation } : {}),
    ...(entity.position ? { position: freezeVec3(entity.position) } : {})
  })
}

function freezeSource(source: SourceHealth): SourceHealth {
  return Object.freeze({ ...source })
}

/**
 * A Map-shaped immutable view.  It captures a private Map and intentionally
 * exposes only the ReadonlyMap surface, so a renderer cannot mutate published
 * data through an accidental cast to Map.
 */
function frozenMap(entries: readonly (readonly [EntityKey, LunarEntity])[]): ReadonlyMap<EntityKey, LunarEntity> {
  const map = new Map(entries)
  let view!: ReadonlyMap<EntityKey, LunarEntity>

  view = Object.freeze({
    [Symbol.iterator]: () => map[Symbol.iterator](),
    entries: () => map.entries(),
    forEach: (callback: (value: LunarEntity, key: EntityKey, map: ReadonlyMap<EntityKey, LunarEntity>) => void) =>
      map.forEach((value, key) => callback(value, key, view)),
    get: (key: EntityKey) => map.get(key),
    get size() {
      return map.size
    },
    has: (key: EntityKey) => map.has(key),
    keys: () => map.keys(),
    values: () => map.values()
  } satisfies ReadonlyMap<EntityKey, LunarEntity>) as ReadonlyMap<EntityKey, LunarEntity>

  return view
}

function snapshot(
  revision: number,
  observedAt: number,
  entries: readonly (readonly [EntityKey, LunarEntity])[],
  sources: readonly SourceHealth[]
): LunarCitySnapshot {
  return Object.freeze({
    entities: frozenMap(entries),
    observedAt,
    revision,
    sources: Object.freeze(sources.map(freezeSource))
  })
}

export function createLunarCitySnapshot(seed: SnapshotSeed = {}): LunarCitySnapshot {
  return snapshot(seed.revision ?? 0, seed.observedAt ?? 0, [], [])
}

export const $lunarCitySnapshot = atom<LunarCitySnapshot>(createLunarCitySnapshot())

/**
 * Clones and freezes a presentation delta before it crosses from adapters to
 * a renderer or any other publisher.  It deliberately excludes mutable input
 * references so callers cannot change a published value after handoff.
 */
export function freezeLunarDelta(delta: LunarDelta): LunarDelta {
  return Object.freeze({
    observedAt: delta.observedAt,
    removals: Object.freeze([...delta.removals]),
    revision: delta.revision,
    sources: Object.freeze(delta.sources.map(freezeSource)),
    upserts: Object.freeze(delta.upserts.map(freezeEntity))
  })
}

/**
 * Applies adapter-owned publication revisions with copy-on-write records. A
 * duplicate or out-of-order publication is a no-op, preserving semantic atom
 * identity for Babylon consumers.
 */
export function applyLunarDelta(delta: LunarDelta): LunarCitySnapshot {
  const frozen = freezeLunarDelta(delta)
  const previous = $lunarCitySnapshot.get()

  if (frozen.revision <= previous.revision) {
    return previous
  }

  const entities = new Map(previous.entities)

  for (const key of frozen.removals) {
    entities.delete(key)
  }

  for (const entity of frozen.upserts) {
    entities.set(entity.key, entity)
  }

  const next = snapshot(frozen.revision, frozen.observedAt, [...entities], frozen.sources)
  $lunarCitySnapshot.set(next)

  return next
}
