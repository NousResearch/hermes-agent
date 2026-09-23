import { entityKey } from './identity'
import type { EntityKey, LunarCitySnapshot, LunarEntity, Vec3 } from './model'

export interface DelegationWorkerPresentation {
  position: Vec3
  moving: boolean
  animation: string
}
export interface DelegationPoint {
  x: number
  y: number
}
export interface DelegationLink {
  parentKey: EntityKey
  childKey: EntityKey
  kind: 'traveling' | 'handoff' | 'return'
  from: DelegationPoint
  to: DelegationPoint
}
export interface DelegationEnvironment {
  presentation(key: EntityKey): DelegationWorkerPresentation | undefined
  project(position: Vec3): DelegationPoint | undefined
}
interface Edge {
  parent: LunarEntity
  child: LunarEntity
  returningMs: number
  handoffMs: number
}
const completed = (entity: LunarEntity) => ['done', 'completed'].includes(entity.sourceState?.toLowerCase() ?? '')

const handoff = (entity: LunarEntity) =>
  ['dependency', 'orchestration', 'handoff'].includes(entity.sourceState?.toLowerCase() ?? '')

/** Presentation of exact observed delegation only; a return line is not an artifact transfer. */
export function createWorkerDelegationController() {
  let edges = new Map<EntityKey, Edge>()

  return {
    update(snapshot: LunarCitySnapshot) {
      const next = new Map<EntityKey, Edge>()

      for (const child of snapshot.entities.values()) {
        if (child.identity.kind !== 'subagent' || child.authority !== 'authoritative') {
          continue
        }

        const identity = child.identity

        const parent = snapshot.entities.get(
          entityKey({
            kind: 'session',
            connectionId: identity.connectionId,
            profile: identity.profile,
            sessionId: identity.sessionId
          })
        )

        if (!parent || parent.authority !== 'authoritative' || completed(parent)) {
          continue
        }

        const previous = edges.get(child.key)

        if (
          previous &&
          (child.observedAt < previous.child.observedAt ||
            parent.observedAt < previous.parent.observedAt ||
            (child.observedAt === previous.child.observedAt && child.sourceState !== previous.child.sourceState) ||
            (parent.observedAt === previous.parent.observedAt && parent.sourceState !== previous.parent.sourceState))
        ) {
          next.set(child.key, previous)

          continue
        }

        const returningMs = completed(child)
          ? previous && child.observedAt > previous.child.observedAt && !completed(previous.child)
            ? 3000
            : (previous?.returningMs ?? 0)
          : 0

        const handoffObserved = handoff(child) || handoff(parent)

        const newHandoff =
          !previous ||
          (child.observedAt > previous.child.observedAt && handoff(child) && !handoff(previous.child)) ||
          (parent.observedAt > previous.parent.observedAt && handoff(parent) && !handoff(previous.parent))

        const handoffMs = handoffObserved ? (newHandoff ? 3000 : (previous?.handoffMs ?? 0)) : 0

        next.set(child.key, { parent, child, returningMs, handoffMs })
      }

      edges = next
    },
    hasCandidates() {
      return [...edges.values()].some(
        edge =>
          edge.returningMs > 0 ||
          edge.handoffMs > 0 ||
          (!completed(edge.child) &&
            !handoff(edge.child) &&
            !handoff(edge.parent) &&
            edge.parent.destination === edge.child.destination &&
            !['unknown', 'unavailable'].includes(edge.parent.destination))
      )
    },
    links(elapsedMs: number, env: DelegationEnvironment): readonly DelegationLink[] {
      const links: DelegationLink[] = []

      for (const edge of [...edges.values()].sort((a, b) => a.child.key.localeCompare(b.child.key))) {
        const elapsed = Number.isFinite(elapsedMs) ? Math.max(0, elapsedMs) : 0
        edge.returningMs = Math.max(0, edge.returningMs - elapsed)
        edge.handoffMs = Math.max(0, edge.handoffMs - elapsed)

        if (links.length >= 8) {
          continue
        }

        const parent = env.presentation(edge.parent.key),
          child = env.presentation(edge.child.key)

        if (!parent || !child) {
          continue
        }

        const kind =
          edge.returningMs > 0
            ? 'return'
            : completed(edge.child)
              ? undefined
              : handoff(edge.child) || handoff(edge.parent)
                ? edge.handoffMs > 0
                  ? 'handoff'
                  : undefined
                : parent.moving &&
                    child.moving &&
                    edge.parent.destination === edge.child.destination &&
                    !['unknown', 'unavailable'].includes(edge.parent.destination)
                  ? 'traveling'
                  : undefined

        if (!kind) {
          continue
        }

        const from = env.project(kind === 'return' ? child.position : parent.position)
        const to = env.project(kind === 'return' ? parent.position : child.position)

        if (!from || !to || ![from.x, from.y, to.x, to.y].every(Number.isFinite)) {
          continue
        }

        links.push({ parentKey: edge.parent.key, childKey: edge.child.key, kind, from, to })
      }

      return links
    },
    clear() {
      edges.clear()
    }
  }
}
