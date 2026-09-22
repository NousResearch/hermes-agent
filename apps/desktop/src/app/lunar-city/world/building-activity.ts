import type { DestinationId, EntityKey, LunarCitySnapshot } from '../model'

interface BuildingClip {
  name: string
  start(loop?: boolean): void
  stop(): void
  reset?(): void
  isPlaying?: boolean
  hasMotion?: boolean
}
interface ActivityRule {
  model: string
  destination: DestinationId
  states: readonly string[]
  clip: string
}

// Source activity selects a decorative prop; it does not establish completion, door access or progress.
// Bus `arrive` currently only rocks the body (generic stateClips), not an arrival choreography.
// Admit it here only once authored metadata establishes the intended arrival action.
const ARRIVALS: Partial<Record<DestinationId, { model: string; clip: string }>> = {
  triage: { model: 'triage', clip: 'door-open' }
}

const RULES: readonly ActivityRule[] = [
  { model: 'review-office', destination: 'review', states: ['review'], clip: 'portal-idle' },
  { model: 'triage', destination: 'triage', states: ['triage', 'blocked', 'failed'], clip: 'triage-station-idle' },
  { model: 'depot', destination: 'depot', states: ['resource_wait', 'waiting_for_resource'], clip: 'lights-idle' },
  { model: 'council', destination: 'council', states: ['dependency', 'orchestration'], clip: 'lights-idle' },
  { model: 'depot', destination: 'depot', states: ['working', 'running'], clip: 'workbench-cycle' },
  { model: 'research-lab', destination: 'lab', states: ['working', 'running'], clip: 'telescope-scan' },
  { model: 'library', destination: 'library', states: ['working', 'running'], clip: 'lights-idle' },
  { model: 'garden', destination: 'garden', states: ['idle', 'heartbeat', 'recovery', 'paused'], clip: 'garden-idle' },
  { model: 'bus', destination: 'bus', states: ['queued', 'ready'], clip: 'idle' }
]

function importedClip(value: unknown): value is BuildingClip {
  if (!value || typeof value !== 'object') {
    return false
  }

  const clip = value as Partial<BuildingClip>

  return typeof clip.name === 'string' && typeof clip.start === 'function' && typeof clip.stop === 'function'
}

/** Scene owns disposal of imports. This controller owns only their playback. */
export function createBuildingActivity() {
  const groups = new Map<string, Map<string, BuildingClip>>()
  const active = new Map<string, BuildingClip>()
  const arrivals = new Map<string, { clip: BuildingClip; entityKey: EntityKey; destination: DestinationId }>()
  const seenArrivals = new Map<EntityKey, string>()
  let snapshot: LunarCitySnapshot | undefined
  let enabled = true
  let reduced = false
  let disposed = false

  function park(clip: BuildingClip) {
    clip.stop()
    clip.reset?.()
  }

  function reconcile() {
    const desired = new Map<string, BuildingClip>()

    for (const [model, arrival] of arrivals) {
      const entity = snapshot?.entities.get(arrival.entityKey)

      if (disposed || !enabled || reduced || entity?.authority !== 'authoritative' || entity.destination !== arrival.destination ||
        (active.get(model) === arrival.clip && !arrival.clip.isPlaying)) {
        arrivals.delete(model)
      } else {
        desired.set(model, arrival.clip)
      }
    }

    if (!disposed && enabled && !reduced && snapshot) {
      const observed = new Set<string>()

      for (const entity of snapshot.entities.values()) {
        if (entity.authority !== 'authoritative' || !entity.sourceState) {
          continue
        }

        const state = entity.sourceState
          .trim()
          .toLowerCase()
          .replace(/[\s-]+/gu, '_')

        observed.add(`${entity.destination}:${state}`)
      }

      for (const rule of RULES) {
        if (desired.size >= 4) {
          break
        }

        const clip = groups.get(rule.model)?.get(rule.clip)

        if (
          clip &&
          clip.hasMotion !== false &&
          !desired.has(rule.model) &&
          rule.states.some(state => observed.has(`${rule.destination}:${state}`))
        ) {
          desired.set(rule.model, clip)
        }
      }
    }

    for (const [model, clip] of active) {
      if (desired.get(model) !== clip) {
        park(clip)
        active.delete(model)
      }
    }

    for (const [model, clip] of desired) {
      if (active.get(model) !== clip) {
        clip.start(!arrivals.has(model))
        active.set(model, clip)
      }
    }
  }

  return {
    register(modelId: string, animationGroups: readonly unknown[]) {
      for (const clip of groups.get(modelId)?.values() ?? []) {
        park(clip)
      }

      active.delete(modelId)
      arrivals.delete(modelId)
      const imported = new Map<string, BuildingClip>()

      for (const candidate of animationGroups) {
        // Stop loader autoplay even when a group is not a supported activity clip.
        if (candidate && typeof candidate === 'object' && 'stop' in candidate && typeof candidate.stop === 'function') {
          candidate.stop()

          if ('reset' in candidate && typeof candidate.reset === 'function') {
            candidate.reset()
          }
        }

        if (importedClip(candidate)) {
          imported.set(candidate.name, candidate)
        }
      }

      if (disposed) {
        return
      }

      groups.set(modelId, imported)
      reconcile()
    },
    /** Caller supplies only a navigation moving-to-settled transition, never a source-state guess. */
    arrival(entityKey: EntityKey, destination: DestinationId) {
      const entity = snapshot?.entities.get(entityKey)
      const rule = ARRIVALS[destination]
      const clip = rule ? groups.get(rule.model)?.get(rule.clip) : undefined

      if (disposed || !enabled || reduced || entity?.authority !== 'authoritative' || entity.destination !== destination ||
        !rule || !clip || clip.hasMotion === false || typeof clip.isPlaying !== 'boolean') {return false}

      const observation = JSON.stringify([destination, entity.observedAt])

      if (seenArrivals.get(entityKey) === observation || arrivals.has(rule.model)) {return false}
      seenArrivals.set(entityKey, observation)
      arrivals.set(rule.model, { clip, entityKey, destination })
      reconcile()

      return true
    },
    /** Call after scene animation advancement; real imported playback completion restores ambient. */
    tick() {
      if (arrivals.size) {reconcile()}

      return active.size > 0
    },
    update(value: LunarCitySnapshot) {
      snapshot = value

      for (const key of seenArrivals.keys()) { if (!value.entities.has(key)) {seenArrivals.delete(key)} }
      reconcile()
    },
    setEnabled(value: boolean) {
      enabled = value
      reconcile()
    },
    setReducedMotion(value: boolean) {
      reduced = value
      reconcile()
    },
    isActive() {
      return active.size > 0
    },
    activeCount() {
      return active.size
    },
    dispose() {
      disposed = true

      for (const clips of groups.values()) {
        for (const clip of clips.values()) {
          park(clip)
        }
      }

      arrivals.clear()
      seenArrivals.clear()
      groups.clear()
      active.clear()
      snapshot = undefined
    }
  }
}
