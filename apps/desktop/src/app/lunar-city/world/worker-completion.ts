import type { EntityKey, LunarCitySnapshot, LunarEntity } from '../model'
import type { WorkerSocialPose } from '../worker-social'

export interface WorkerCompletionEnvironment {
  enabled: boolean
  available(key: EntityKey): boolean
  moving(key: EntityKey): boolean
  /** Actual playable clips on this worker, including motion validation where available. */
  supports(key: EntityKey, clip: string): boolean
}
interface Completion { entity: LunarEntity; remainingMs: number; clip?: string }
const completed = (entity: LunarEntity) => ['done', 'completed'].includes(entity.sourceState?.trim().toLowerCase() ?? '')
const finiteCompletion = new Set(['done', 'handoff', 'celebrate'])

/** A short completion acknowledgement, never a carried artifact or a change to operational state. */
export function createWorkerCompletionController() {
  let observed = new Map<EntityKey, LunarEntity>()
  const pending = new Map<EntityKey, Completion>()

  return {
    update(snapshot: LunarCitySnapshot) {
      const next = new Map<EntityKey, LunarEntity>()

      for (const entity of snapshot.entities.values()) {
        if (entity.authority !== 'authoritative') {continue}
        const previous = observed.get(entity.key)
        // Out-of-order authoritative polls cannot roll history back and replay a completion.
        const latest = previous && entity.observedAt < previous.observedAt ? previous : entity
        next.set(entity.key, latest)

        if (latest !== entity) {continue}

        if (!completed(entity)) { pending.delete(entity.key);

 continue }

        if (previous?.sourceState && !completed(previous) && Number.isFinite(entity.observedAt) &&
          Number.isFinite(previous.observedAt) && entity.observedAt > previous.observedAt) {
          pending.set(entity.key, { entity, remainingMs: 3000 })
        } else {
          const active = pending.get(entity.key)

          if (active) {active.entity = entity}
        }
      }

      for (const key of pending.keys()) {if (!next.has(key)) {pending.delete(key)}}
      observed = next
    },
    tick(elapsedMs: number, env: WorkerCompletionEnvironment): ReadonlyMap<EntityKey, WorkerSocialPose> {
      const poses = new Map<EntityKey, WorkerSocialPose>()

      if (!env.enabled) { pending.clear();

 return poses }

      for (const [key, completion] of pending) {
        completion.remainingMs -= Math.max(0, elapsedMs)

        if (completion.remainingMs <= 0 || !env.available(key) || env.moving(key) ||
          (finiteCompletion.has(completion.entity.animation) && env.supports(key, completion.entity.animation))) {
          pending.delete(key)

          continue
        }

        const clip = completion.clip ?? ['handoff', 'celebrate'].find(candidate => env.supports(key, candidate))

        if (!clip || !env.supports(key, clip) || poses.size >= 4) { pending.delete(key);

 continue }

        completion.clip = clip
        poses.set(key, { animation: clip })
      }

      return poses
    },
    isActive() { return pending.size > 0 },
    clear() { pending.clear(); observed.clear() }
  }
}
