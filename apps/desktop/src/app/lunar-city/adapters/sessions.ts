import { sessionOwnerRouteFromRow } from '@/store/session-request-router'
import type { SubagentProgress } from '@/store/subagents'
import type { SessionInfo } from '@/types/hermes'

import { entityKey } from '../identity'
import type { LunarEntity, SourceHealth } from '../model'
import { mapObservedState } from '../state-map'

export interface SessionObservation {
  fresh?: boolean
  legacyConnectionId?: string
  legacySingleBackend?: boolean
  observedAt: number
  sourceObservations?: ReadonlyMap<string, SessionSourceObservation>
}

export interface SessionSourceObservation {
  /** Monotonic source generation. It is provenance, never wall-clock freshness. */
  generation: number
  fresh: boolean
  observedAt: number
}

export interface NormalizedSessions {
  entities: readonly LunarEntity[]
  sources: readonly SourceHealth[]
}

export interface NormalizedSubagents {
  entities: readonly LunarEntity[]
}

function isFresh(observation: SessionObservation): boolean {
  return observation.fresh !== false
}

function sourceObservation(
  connectionId: string,
  profile: string,
  observation: Pick<SessionObservation, 'fresh' | 'observedAt' | 'sourceObservations'>
): Pick<SessionSourceObservation, 'fresh' | 'observedAt'> {
  const source = observation.sourceObservations?.get(ownerObservationKey(connectionId, profile))

  return source ?? { fresh: observation.fresh !== false, observedAt: observation.observedAt }
}

export function ownerObservationKey(connectionId: string, profile: string): string {
  return JSON.stringify([connectionId, profile])
}

export function sessionSourceName(connectionId: string, profile: string): string {
  return `session:${encodeURIComponent(connectionId)}:${encodeURIComponent(profile)}`
}

export function delegationSourceName(connectionId: string, profile: string): string {
  return `delegation:${encodeURIComponent(connectionId)}:${encodeURIComponent(profile)}`
}

function exactSessionOwner(
  row: SessionInfo,
  observation: SessionObservation
): { connectionId: string; profile: string } | undefined {
  const routed = sessionOwnerRouteFromRow(row)

  if (routed) {
    return routed
  }

  if (!observation.legacySingleBackend || !observation.legacyConnectionId?.trim()) {
    return undefined
  }

  return { connectionId: observation.legacyConnectionId.trim(), profile: row.profile?.trim() || 'default' }
}

function sessionStatus(row: SessionInfo): string {
  if (row.ended_at !== null && row.ended_at !== undefined) {
    return 'completed'
  }

  return row.is_active ? 'running' : 'idle'
}

function sourceRows(entities: readonly LunarEntity[]): readonly SourceHealth[] {
  const sources = new Map<string, SourceHealth>()

  for (const entity of entities) {
    const source = sessionSourceName(entity.identity.connectionId, entity.identity.profile)
    const prior = sources.get(source)

    if (prior && (prior.authority !== entity.authority || prior.observedAt !== entity.observedAt)) {
      // One connection source must have one canonical observation. A mixed
      // batch is partial rather than borrowing freshness from either row.
      sources.set(source, {
        authority: 'partial',
        observedAt: Math.min(prior.observedAt, entity.observedAt),
        source
      })
    } else if (!prior) {
      sources.set(source, { authority: entity.authority, observedAt: entity.observedAt, source })
    }
  }

  return [...sources.values()].sort((left, right) => left.source.localeCompare(right.source))
}

/**
 * Normalizes session rows only when their connection/profile owner is explicit.
 * A legacy connection can be supplied only by a caller that has proven the
 * single-backend topology; active UI state is deliberately never consulted.
 */
export function normalizeSessions(rows: readonly SessionInfo[], observation: SessionObservation): NormalizedSessions {
  const entities = rows
    .flatMap(row => {
      const owner = exactSessionOwner(row, observation)

      if (!owner || !row.id.trim()) {
        return []
      }

      const identity = {
        connectionId: owner.connectionId,
        kind: 'session' as const,
        profile: owner.profile,
        sessionId: row.id
      }

      const source = sourceObservation(owner.connectionId, owner.profile, observation)
      const state = mapObservedState({ fresh: isFresh(source), source: 'session', status: sessionStatus(row) })

      return [
        {
          ...state,
          identity,
          key: entityKey(identity),
          observedAt: source.observedAt,
          ...(row.git_repo_root?.trim() ? { projectId: row.git_repo_root } : {})
        } satisfies LunarEntity
      ]
    })
    .sort((left, right) => left.key.localeCompare(right.key))

  return { entities, sources: sourceRows(entities) }
}

function statusForSubagent(status: SubagentProgress['status'] | string): string {
  if (status === 'complete') {
    return 'completed'
  }

  return status
}

/** Converts one bounded progress row into a presentation entity. */
export function normalizeSubagent(
  row: Pick<SubagentProgress, 'id' | 'status'>,
  owner: Extract<LunarEntity['identity'], { kind: 'session' }>,
  observation: Pick<SessionObservation, 'fresh' | 'observedAt'>
): LunarEntity | undefined {
  if (!row.id.trim()) {
    return undefined
  }

  const identity = {
    connectionId: owner.connectionId,
    kind: 'subagent' as const,
    profile: owner.profile,
    sessionId: owner.sessionId,
    subagentId: row.id
  }

  const state = mapObservedState({
    fresh: observation.fresh !== false,
    source: 'subagent',
    status: statusForSubagent(row.status)
  })

  return { ...state, identity, key: entityKey(identity), observedAt: observation.observedAt }
}

export interface OwnedSubagentRows {
  owner: Extract<LunarEntity['identity'], { kind: 'session' }>
  rows: readonly SubagentProgress[]
}

export interface OwnedSubagentLimits {
  maxOwners: number
  maxRows: number
  maxRowsPerOwner: number
}

const DEFAULT_OWNED_SUBAGENT_LIMITS: OwnedSubagentLimits = Object.freeze({
  maxOwners: 64,
  maxRows: 512,
  maxRowsPerOwner: 64
})

function conflictingEntity(entity: LunarEntity): LunarEntity {
  return { ...entity, animation: 'unavailable', authority: 'partial', destination: 'unknown' }
}

/**
 * Canonical bounded merge for callers that already retain exact parent
 * ownership. Duplicate parent/child ids on other connections remain distinct.
 */
export function normalizeOwnedSubagents(
  batches: readonly OwnedSubagentRows[],
  observation: Pick<SessionObservation, 'fresh' | 'observedAt' | 'sourceObservations'>,
  limits: OwnedSubagentLimits = DEFAULT_OWNED_SUBAGENT_LIMITS
): NormalizedSubagents {
  const totalRows = batches.reduce((total, batch) => total + batch.rows.length, 0)

  if (
    batches.length > limits.maxOwners ||
    totalRows > limits.maxRows ||
    batches.some(batch => batch.rows.length > limits.maxRowsPerOwner)
  ) {
    return { entities: [] }
  }

  const entities = new Map<string, LunarEntity>()

  for (const batch of batches) {
    const source = sourceObservation(batch.owner.connectionId, batch.owner.profile, observation)

    for (const row of batch.rows) {
      if (row.parentId !== null && row.parentId !== batch.owner.sessionId) {
        continue
      }

      const entity = normalizeSubagent(row, batch.owner, source)

      if (entity) {
        const prior = entities.get(entity.key)

        if (!prior) {
          entities.set(entity.key, entity)
        } else if (
          prior.animation !== entity.animation ||
          prior.authority !== entity.authority ||
          prior.destination !== entity.destination ||
          prior.observedAt !== entity.observedAt
        ) {
          entities.set(entity.key, conflictingEntity(prior))
        }
      }
    }
  }

  return { entities: [...entities.values()].sort((left, right) => left.key.localeCompare(right.key)) }
}

/**
 * The subagent store is keyed only by parent session id.  A parent id that
 * resolves to zero or more than one exact session owner is unrepresentable in
 * the city rather than being guessed onto the active source.
 */
export function normalizeSubagents(
  byParentSession: Readonly<Record<string, readonly SubagentProgress[]>>,
  parentSessions: readonly LunarEntity[],
  observation: Pick<SessionObservation, 'fresh' | 'observedAt' | 'sourceObservations'>
): NormalizedSubagents {
  const owners = new Map<string, Extract<LunarEntity['identity'], { kind: 'session' }>[]>()

  for (const entity of parentSessions) {
    if (entity.identity.kind !== 'session') {
      continue
    }

    const candidates = owners.get(entity.identity.sessionId) ?? []
    candidates.push(entity.identity)
    owners.set(entity.identity.sessionId, candidates)
  }

  const entities = Object.entries(byParentSession)
    .flatMap(([parentSessionId, rows]) => {
      const candidates = owners.get(parentSessionId)

      if (candidates?.length !== 1) {
        return []
      }

      const owner = candidates[0]!
      const source = sourceObservation(owner.connectionId, owner.profile, observation)

      return rows.flatMap(row => {
        if (row.parentId !== null && row.parentId !== parentSessionId) {
          return []
        }

        const entity = normalizeSubagent(row, owner, source)

        return entity ? [entity] : []
      })
    })
    .sort((left, right) => left.key.localeCompare(right.key))

  return { entities }
}
