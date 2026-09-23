import { entityKey } from './identity'
import type { EntityKey, LunarCitySnapshot, LunarEntity, Vec3 } from './model'
import { deriveWorkerPersonality } from './worker-personality'

export interface WorkerSocialPose { animation: string; facing?: number }
export interface WorkerEncounter {
  id: string
  participants: readonly [EntityKey, EntityKey]
  phase: 'greeting' | 'exchange' | 'farewell'
  elapsedMs: number
  provenance: 'local-ambient'
}
export interface WorkerSocialEnvironment {
  enabled: boolean
  supports(key: EntityKey, animation: string): boolean
  position(key: EntityKey): Vec3 | undefined
  available(key: EntityKey): boolean
}
const REST_STATES = new Set(['idle', 'rest', 'paused', 'pause', 'recovery'])

export function socialEligible(entity: LunarEntity): boolean {
  return entity.authority === 'authoritative' && entity.destination === 'garden' &&
    REST_STATES.has((entity.sourceState ?? entity.animation).toLowerCase())
}

/** Exact upstream relationships only; shared group membership is not delegated work. */
export function workerRelationships(entity: LunarEntity, snapshot: LunarCitySnapshot): readonly LunarEntity[] {
  const owner = entity.identity

  if (owner.kind === 'subagent') {
    const parent = snapshot.entities.get(entityKey({ kind: 'session', connectionId: owner.connectionId,
      profile: owner.profile, sessionId: owner.sessionId }))

    return parent ? [parent] : []
  }

  if (owner.kind !== 'session') {return []}

  return [...snapshot.entities.values()].filter(candidate => candidate.identity.kind === 'subagent' &&
    candidate.identity.connectionId === owner.connectionId && candidate.identity.profile === owner.profile &&
    candidate.identity.sessionId === owner.sessionId).sort((a, b) => a.key.localeCompare(b.key))
}

/** Local gestures never publish snapshots, send speech, or change task state. */
export function createWorkerSocialController() {
  let entities: ReadonlyMap<EntityKey, LunarEntity> = new Map()
  const encounters = new Map<string, WorkerEncounter>()
  const cooldowns = new Map<EntityKey, number>()
  const previews = new Map<EntityKey, { animation: string; remainingMs: number }>()
  let clockMs = 0
  let sequence = 0
  let autoAtMs = 0
  const poses = new Map<EntityKey, WorkerSocialPose>()
  const busy = (key: EntityKey) => previews.has(key) || [...encounters.values()].some(e => e.participants.includes(key))

  const eligible = (key: EntityKey, env: WorkerSocialEnvironment) => {
    const entity = entities.get(key)

    return entity && socialEligible(entity) && env.available(key) && env.position(key) !== undefined
  }

  const partner = (key: EntityKey, env: WorkerSocialEnvironment): EntityKey | undefined => {
    const self = entities.get(key), origin = env.position(key)

    if (!self || !origin) {return undefined}

    return [...entities.values()].filter(other => other.key !== key &&
      other.identity.connectionId === self.identity.connectionId && env.supports(other.key,'talk') && env.supports(other.key,'listen') && eligible(other.key, env) && !busy(other.key) &&
      (cooldowns.get(other.key) ?? 0) <= clockMs).sort((a,b) => a.key.localeCompare(b.key))
      .find(other => { const p = env.position(other.key)!; const d = Math.hypot(p.x-origin.x,p.y-origin.y,p.z-origin.z)

        return d >= .7 && d <= 6 })?.key
  }

  const begin = (key: EntityKey, env: WorkerSocialEnvironment): boolean => {
    if (!env.enabled || !env.supports(key,'talk') || !env.supports(key,'listen') || encounters.size >= 2 || !eligible(key,env) || busy(key) || (cooldowns.get(key) ?? 0)>clockMs) {return false}
    const other = partner(key,env)

    if (!other) {return false}
    const id = `local-encounter:${++sequence}`
    encounters.set(id,{id,participants:[key,other],phase:'greeting',elapsedMs:0,provenance:'local-ambient'})

    return true
  }

  return {
    update(snapshot: LunarCitySnapshot) {
      entities = snapshot.entities

      for (const key of cooldowns.keys()) {if (!entities.has(key)) {cooldowns.delete(key)}}
    },
    requestGreeting(key: EntityKey, env: WorkerSocialEnvironment): boolean { return begin(key,env) },
    requestGesture(key: EntityKey, animation: string, env: WorkerSocialEnvironment): boolean {
      if (!env.enabled || !env.supports(key,animation === 'inspect' ? 'think' : animation) || !eligible(key,env) || busy(key) || !['talk','heartbeat','think','inspect'].includes(animation)) {return false}
      previews.set(key,{animation:animation === 'inspect' ? 'think' : animation,remainingMs:3000})

      return true
    },
    tick(elapsedMs: number, env: WorkerSocialEnvironment): ReadonlyMap<EntityKey,WorkerSocialPose> {
      clockMs += Math.max(0,elapsedMs); poses.clear()

      if (!env.enabled) { encounters.clear(); previews.clear();

 return poses }

      for (const [key,preview] of previews) {
        preview.remainingMs -= elapsedMs

        if (!eligible(key,env) || preview.remainingMs<=0) {previews.delete(key)}
        else {poses.set(key,{animation:preview.animation})}
      }

      for (const [id,encounter] of encounters) {
        const [a,b] = encounter.participants, pa = env.position(a), pb = env.position(b)

        if (!eligible(a,env) || !eligible(b,env) || !pa || !pb || Math.hypot(pa.x-pb.x,pa.z-pb.z)>6) {
          encounters.delete(id);

 for(const key of [a,b]){cooldowns.set(key,clockMs+15000);}

 continue
        }

        const elapsed = encounter.elapsedMs + elapsedMs

        if(elapsed>=9000) { encounters.delete(id);

 for(const key of [a,b]){cooldowns.set(key,clockMs+deriveWorkerPersonality(entities.get(key)!).cadenceSeconds*1000);}

 continue }

        const phase = elapsed<2000?'greeting':elapsed<7000?'exchange':'farewell'
        encounters.set(id,{...encounter,elapsedMs:elapsed,phase})
        const firstTalking = elapsed<4500
        poses.set(a,{animation:firstTalking?'talk':'listen',facing:Math.atan2(pb.x-pa.x,pb.z-pa.z)})
        poses.set(b,{animation:firstTalking?'listen':'talk',facing:Math.atan2(pa.x-pb.x,pa.z-pb.z)})
      }

      if(clockMs>=autoAtMs) {
        autoAtMs=clockMs+12000

        for(const entity of [...entities.values()].filter(entity=>eligible(entity.key,env)).sort((a,b)=>a.key.localeCompare(b.key)).slice(0,64)) {
          if(encounters.size>=2){break}
          begin(entity.key,env)
        }
      }

      return poses
    },
    nextWakeDelay(env: WorkerSocialEnvironment): number | undefined {
      return env.enabled && [...entities.keys()].some(key=>eligible(key,env)) ? Math.max(1,autoAtMs-clockMs) : undefined
    },
    encounters(): readonly WorkerEncounter[] { return [...encounters.values()] },
    active(): boolean { return encounters.size>0 || previews.size>0 }
  }
}
