import { entityKey, projectCompoundKey } from '../identity'
import type { EntityKey, LunarCitySnapshot, LunarEntity, ProjectSlotManifestEntry, Vec3 } from '../model'

export interface ProjectCompoundAnchor {
  connectionId: string
  key: string
  position: Vec3
  projectId: string
}

export interface ProjectCompoundSummary {
  connectionId: string
  key: string
  projectId: string
  entityKeys: readonly EntityKey[]
  totalCount: number
  authoritativeCount: number
  workCount: number
  slotId?: string
  position?: Vec3
  unplaced: boolean
}

export interface ProjectCompoundReport {
  compounds: readonly ProjectCompoundSummary[]
  anchors: readonly ProjectCompoundAnchor[]
  entityTargets: ReadonlyMap<EntityKey, Vec3>
  overflowCount: number
  slotIssues: readonly string[]
}

function overlappingSlots(a: ProjectSlotManifestEntry, b: ProjectSlotManifestEntry): boolean {
  return a.bounds.min.x < b.bounds.max.x && a.bounds.max.x > b.bounds.min.x &&
    a.bounds.min.z < b.bounds.max.z && a.bounds.max.z > b.bounds.min.z
}

/** Stable, bounded presentation allocation. Project IDs are opaque source IDs, never normalized paths. */
export function createProjectCompoundController(manifestSlots: readonly ProjectSlotManifestEntry[]) {
  const slots: ProjectSlotManifestEntry[] = []
  const slotIssues: string[] = []

  for (const slot of manifestSlots) {
    if (slots.some(other => other.id === slot.id || overlappingSlots(other, slot))) {
      slotIssues.push(`Slot ${slot.id} duplicates an ID or overlaps an earlier manifest footprint; excluded.`)
    } else {slots.push(slot)}
  }

  const retained = new Map<string, string>()

  return {
    update(snapshot: LunarCitySnapshot): ProjectCompoundReport {
      const inputs = new Map<string, { connectionId: string; projectId: string; entities: LunarEntity[] }>()

      for (const entity of snapshot.entities.values()) {
        if (entity.identity.kind === 'profile') {continue}
        let projectId = entity.projectId

        if (!projectId?.trim() && entity.identity.kind === 'subagent') {
          const identity = entity.identity

          const parent = snapshot.entities.get(entityKey({ kind: 'session', connectionId: identity.connectionId,
            profile: identity.profile, sessionId: identity.sessionId }))

          projectId = parent?.projectId
        }

        if (!projectId?.trim()) {continue}
        const key = projectCompoundKey(entity.identity.connectionId, projectId)
        const group = inputs.get(key) ?? { connectionId: entity.identity.connectionId, projectId, entities: [] }
        group.entities.push(entity)
        inputs.set(key, group)
      }

      for (const key of retained.keys()) {if (!inputs.has(key)) {retained.delete(key)}}
      const usedSlots = new Set(retained.values())
      const sorted = [...inputs.entries()].sort(([a], [b]) => a.localeCompare(b))

      for (const [key] of sorted) {
        if (retained.has(key)) {continue}
        const free = slots.find(slot => !usedSlots.has(slot.id))

        if (free) { retained.set(key, free.id); usedSlots.add(free.id) }
      }

      const anchors: ProjectCompoundAnchor[] = []
      const entityTargets = new Map<EntityKey, Vec3>()

      const compounds = sorted.map(([key, input]): ProjectCompoundSummary => {
        const slot = slots.find(candidate => candidate.id === retained.get(key))
        const authoritative = input.entities.filter(entity => entity.authority === 'authoritative')

        if (slot) {
          anchors.push({ key, connectionId: input.connectionId, projectId: input.projectId, position: { ...slot.position } })

          for (const entity of authoritative) {
            if (entity.destination === 'project') {entityTargets.set(entity.key, { ...slot.navigationLink.from })}
          }
        }

        return {
          key, connectionId: input.connectionId, projectId: input.projectId,
          entityKeys: input.entities.map(entity => entity.key).sort(),
          totalCount: input.entities.length,
          authoritativeCount: authoritative.length,
          workCount: authoritative.filter(entity => ['working', 'running'].includes(entity.sourceState?.trim().toLowerCase() ?? '')).length,
          ...(slot ? { slotId: slot.id, position: { ...slot.position } } : {}),
          unplaced: !slot
        }
      })

      return { compounds, anchors, entityTargets, overflowCount: compounds.filter(compound => compound.unplaced).length,
        slotIssues: [...slotIssues] }
    }
  }
}
