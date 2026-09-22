import { entityKey } from './identity'
import type { LunarEntity, LunarGroupMembership, LunarPresentationMetadata } from './model'

export type WorkerGesture = 'inspect' | 'heartbeat' | 'talk'

interface AuthoredPersonality {
  trait: 'Curious' | 'Methodical' | 'Helpful' | 'Expressive'
  description: string
  gesture: WorkerGesture
  cadenceSeconds: number
}

export interface WorkerPersonalityPresentation extends AuthoredPersonality {
  provenance: 'authored-local'
  configuredTitle?: string
  groups: readonly LunarGroupMembership[]
  metadataState: LunarPresentationMetadata['state']
  metadataSource?: string
}

const PERSONALITIES: readonly AuthoredPersonality[] = [
  { trait: 'Curious', description: 'Pauses for a small look around.', gesture: 'inspect', cadenceSeconds: 36 },
  { trait: 'Methodical', description: 'Uses quiet, evenly spaced check-in motions.', gesture: 'heartbeat', cadenceSeconds: 48 },
  { trait: 'Helpful', description: 'Offers an occasional gentle acknowledgement.', gesture: 'heartbeat', cadenceSeconds: 42 },
  { trait: 'Expressive', description: 'Adds a brief conversational gesture.', gesture: 'talk', cadenceSeconds: 32 }
]

/** The profile owner keeps its style across sessions; labels never imply a SOUL or change authority. */
export function deriveWorkerPersonality(entity: LunarEntity): WorkerPersonalityPresentation {
  const owner = entityKey({ kind: 'profile', connectionId: entity.identity.connectionId, profile: entity.identity.profile })
  let seed = 2166136261

  for (const character of owner) {seed = Math.imul(seed ^ character.charCodeAt(0), 16777619) >>> 0}
  const metadata = entity.presentation?.metadata
  const fresh = metadata?.state === 'fresh'

  return {
    ...PERSONALITIES[seed % PERSONALITIES.length],
    provenance: 'authored-local',
    configuredTitle: fresh ? entity.presentation?.configuredTitle : undefined,
    groups: fresh ? entity.presentation?.groups ?? [] : [],
    metadataState: metadata?.state ?? 'unavailable',
    metadataSource: metadata?.source
  }
}
