import type { EntityKey, Vec3 } from '../model'

export interface SavedEntityPosition {
  position: Vec3
  sourcePosition?: Vec3
}

export interface EntityPositionState {
  read(key: EntityKey): SavedEntityPosition | undefined
  /** Only a fresh authoritative observation renews the retention lease. */
  save(key: EntityKey, value: SavedEntityPosition, observed: boolean): void
  forget(key: EntityKey): void
}

interface PositionEntry extends SavedEntityPosition {
  expiresAt: number
}

interface PositionStateOptions {
  /** Include both world and navigation identity; never a display label alone. */
  scope: string
  maxEntries?: number
  ttlMs?: number
  now?: () => number
}

const scopes = new Map<string, Map<EntityKey, PositionEntry>>()
const MAX_SCOPES = 8

function copy(value: SavedEntityPosition): SavedEntityPosition {
  return { position: { ...value.position }, sourcePosition: value.sourcePosition ? { ...value.sourcePosition } : undefined }
}

/** Bounded process-local presentation memory. No source state or browser storage is written. */
export function createEntityPositionState({ scope, maxEntries = 512, ttlMs = 600_000, now = Date.now }: PositionStateOptions): EntityPositionState {
  const capacity = Math.max(1, Math.min(2000, Math.floor(maxEntries)))
  const lifetime = Math.max(1, Math.min(3_600_000, ttlMs))

  let nextSweepAt = 0

  const entries = () => {
    let values = scopes.get(scope)

    if (!values) {
      values = new Map()
      scopes.set(scope, values)

      while (scopes.size > MAX_SCOPES) {scopes.delete(scopes.keys().next().value!)}
    }

    const time = now()

    if (time >= nextSweepAt) {
      for (const [key, value] of values) {if (value.expiresAt <= time) {values.delete(key)}}
      nextSweepAt = time + Math.min(1000, lifetime)
    }

    return values
  }

  return {
    read(key) {
      const value = entries().get(key)

      return value && value.expiresAt > now() ? copy(value) : undefined
    },
    save(key, value, observed) {
      const values = entries()
      const retained = values.get(key)
      const previous = retained && retained.expiresAt > now() ? retained : undefined

      if (!observed && !previous) {return}
      const expiresAt = observed ? now() + lifetime : previous!.expiresAt
      values.delete(key)
      values.set(key, { ...copy(value), expiresAt })

      while (values.size > capacity) {values.delete(values.keys().next().value!)}
    },
    forget(key) { scopes.get(scope)?.delete(key) }
  }
}
