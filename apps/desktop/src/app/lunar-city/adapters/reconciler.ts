import { onGatewayEvent } from '@/contrib/events'
import type { DesktopAgentRoster } from '@/global'
import { $connectionsRegistry, hasRegistryTopology } from '@/store/connection-registry-state'
import { $fleetRoster, refreshFleetRoster } from '@/store/fleet-roster'
import { requestGatewayForAgent } from '@/store/gateway'
import { $connection, $sessions, ownerLookupSessionRows } from '@/store/session'
import { sessionOwnerRouteFromRow } from '@/store/session-request-router'
import { $subagentsBySession, type SubagentProgress } from '@/store/subagents'
import type { SessionInfo } from '@/types/hermes'

import type { LunarEntity, LunarPresentationMetadata, SourceHealth } from '../model'
import { $lunarCitySnapshot, applyLunarDelta, freezeLunarDelta, type LunarDelta } from '../store'

import {
  createBotRosterPlacementState,
  enrichBotRosterEntities,
  type ScopedProfilesListRequest
} from './bot-roster-details'
import { normalizeRoster } from './fleet'
import {
  delegationSourceName,
  normalizeOwnedSubagents,
  normalizeSessions,
  normalizeSubagents,
  type OwnedSubagentRows,
  ownerObservationKey,
  sessionSourceName,
  type SessionSourceObservation
} from './sessions'

export interface ReconcileReadResult {
  /** False means the read is partial and therefore cannot remove prior rows. */
  authoritative?: boolean
  entities: readonly LunarEntity[]
  /** Source-owned rows this read may replace while another source is partial. */
  replacementSources?: readonly string[]
  /** On bounded removal overflow, stale unlisted prior entities only in these source namespaces. */
  staleUnlistedSourcePrefixes?: readonly string[]
  sources: readonly SourceHealth[]
}

export interface ReconcilerEvent {
  sequence?: number
  source?: SequenceSource
}

export interface SequenceSource {
  connectionId: string
  profile: string
  sessionId: string
}

export interface SequenceScope {
  connectionId?: string
  profile?: string
  sessionId?: string
}

export interface FleetRefreshResult {
  error?: string
  observedAt?: number
  status?: 'failed' | 'partial' | 'refreshed' | 'retained'
}

export interface LunarCityOptionalSource {
  read: () => Promise<ReconcileReadResult>
  start?: (listener: () => void) => () => void
}

export interface LunarCityReconcilerOptions {
  freshnessMs?: number
  now?: () => number
  publish?: (delta: LunarDelta) => unknown
  read: () => Promise<ReconcileReadResult>
}

interface ReadableAtom<T> {
  get(): T
  listen(listener: (value: T) => void): () => void
}

export interface LunarCityLiveSources {
  $fleetRoster: ReadableAtom<DesktopAgentRoster | null>
  $sessions?: ReadableAtom<readonly SessionInfo[]>
  $subagentsBySession: ReadableAtom<Readonly<Record<string, readonly SubagentProgress[]>>>
  legacyConnectionId?: () => string | undefined
  legacySingleBackend: () => boolean
  readProfileRoster?: ScopedProfilesListRequest
  readDelegationStatus?: (connectionId: string, profile: string) => Promise<unknown>
  readSessionList?: (connectionId: string, profile: string) => Promise<unknown>
  onEvent?: (listener: (event: unknown) => void) => () => void
  onFocus?: (listener: () => void) => () => void
  onRegistryChange?: (listener: () => void) => () => void
  optionalSources?: readonly LunarCityOptionalSource[]
  refreshFleet: (options: { force: boolean }) => Promise<FleetRefreshResult | void>
  sessionRows?: () => readonly SessionInfo[]
}

export interface StartLunarCityReconcilerOptions {
  freshnessMs?: number
  now?: () => number
  optionalSources?: readonly LunarCityOptionalSource[]
  sources?: LunarCityLiveSources
}

export type ReconcileEventResult = 'accepted' | 'ignored'
const MAX_RETIRED_SESSION_OWNERS = 256

/** A backend gap, not a duplicate or ordinary next event, needs a reread. */
export function shouldReconcile(currentRevision: number, incomingRevision: number): boolean {
  return incomingRevision > currentRevision + 1
}

function sameEntity(left: LunarEntity, right: LunarEntity): boolean {
  return (
    left.key === right.key &&
    left.observedAt === right.observedAt &&
    left.authority === right.authority &&
    left.destination === right.destination &&
    left.animation === right.animation &&
    left.sourceState === right.sourceState &&
    left.projectId === right.projectId &&
    left.variant === right.variant &&
    JSON.stringify(left.presentation) === JSON.stringify(right.presentation) &&
    left.identity.kind === right.identity.kind &&
    left.identity.connectionId === right.identity.connectionId &&
    JSON.stringify(left.identity) === JSON.stringify(right.identity) &&
    JSON.stringify(left.position) === JSON.stringify(right.position)
  )
}

function sameSources(left: readonly SourceHealth[], right: readonly SourceHealth[]): boolean {
  if (left.length !== right.length) {
    return false
  }

  return left.every((source, index) => {
    const other = right[index]

    return (
      other?.source === source.source &&
      other.authority === source.authority &&
      other.observedAt === source.observedAt &&
      other.error === source.error
    )
  })
}

function sortedSources(sources: readonly SourceHealth[]): readonly SourceHealth[] {
  return [...sources].sort((left, right) => left.source.localeCompare(right.source))
}

function sourceKey(source: SequenceSource): string {
  return JSON.stringify([source.connectionId, source.profile, source.sessionId])
}

function belongsToScope(source: SequenceSource, scope: SequenceScope): boolean {
  return (
    (scope.connectionId === undefined || source.connectionId === scope.connectionId) &&
    (scope.profile === undefined || source.profile === scope.profile) &&
    (scope.sessionId === undefined || source.sessionId === scope.sessionId)
  )
}

function healthSourceFor(entity: LunarEntity): string {
  switch (entity.identity.kind) {
    case 'profile':
      return `fleet:${entity.identity.connectionId}`

    case 'session':
      return sessionSourceName(entity.identity.connectionId, entity.identity.profile)

    case 'subagent':
      return delegationSourceName(entity.identity.connectionId, entity.identity.profile)

    case 'kanban':
      return `kanban:${encodeURIComponent(entity.identity.connectionId)}:${encodeURIComponent(entity.identity.profile)}`
  }
}

function staleEntity(entity: LunarEntity): LunarEntity {
  if (entity.authority === 'stale') {
    return entity
  }

  return { ...entity, animation: 'unavailable', authority: 'stale', destination: 'unknown' }
}

/**
 * Event-first bounded reconciliation.  It has no timer or frame polling; the
 * sole timer is the earliest source freshness deadline, and it is disposed
 * together with every late-publication guard.
 */
export class LunarCityReconciler {
  private active = false
  private freshnessTimer: ReturnType<typeof setTimeout> | undefined
  private inFlight = false
  private queued = false
  private readonly sequences = new Map<string, { sequence: number; source: SequenceSource }>()

  constructor(private readonly options: LunarCityReconcilerOptions) {}

  start(): () => void {
    if (!this.active) {
      this.active = true
      this.invalidate('mount')
    }

    return () => this.stop()
  }

  stop(): void {
    this.active = false
    this.queued = false

    if (this.freshnessTimer !== undefined) {
      clearTimeout(this.freshnessTimer)
      this.freshnessTimer = undefined
    }
  }

  acceptEvent(event: ReconcilerEvent): ReconcileEventResult {
    const sequence = event.sequence

    if (sequence === undefined || !Number.isFinite(sequence)) {
      this.invalidate('unsequenced-event')

      return 'accepted'
    }

    if (!event.source) {
      this.invalidate('unscoped-event')

      return 'accepted'
    }

    const key = sourceKey(event.source)
    const prior = this.sequences.get(key)

    if (prior !== undefined && sequence <= prior.sequence) {
      return 'ignored'
    }

    this.sequences.set(key, { sequence, source: event.source })
    this.invalidate(prior !== undefined && shouldReconcile(prior.sequence, sequence) ? 'sequence-gap' : 'event')

    return 'accepted'
  }

  /**
   * A restarted gateway can legitimately replay a lower sequence.  Clear only
   * the exact connection/profile/session scope that declared the restart; an
   * unscoped restart clears all cursors rather than retaining false ordering.
   */
  resetSequences(scope: SequenceScope = {}): void {
    for (const [key, entry] of this.sequences) {
      if (belongsToScope(entry.source, scope)) {
        this.sequences.delete(key)
      }
    }
  }

  invalidate(_reason: string): void {
    if (!this.active) {
      return
    }

    if (this.inFlight) {
      this.queued = true

      return
    }

    void this.readOnce()
  }

  private async readOnce(): Promise<void> {
    if (!this.active || this.inFlight) {
      return
    }

    this.inFlight = true

    try {
      const result = await this.options.read()

      if (!this.active) {
        return
      }

      this.publishRead(result)
    } catch {
      if (this.active) {
        this.publishStale()
      }
    } finally {
      this.inFlight = false

      if (this.active && this.queued) {
        this.queued = false
        void this.readOnce()
      }
    }
  }

  private publishRead(result: ReconcileReadResult): void {
    const previous = $lunarCitySnapshot.get()
    const candidates = result.authoritative === false ? new Map(previous.entities) : new Map<string, LunarEntity>()

    const stalePartialSources = new Set(
      result.authoritative === false
        ? result.sources.filter(source => source.authority === 'stale').map(source => source.source)
        : []
    )

    for (const [key, entity] of candidates) {
      if (stalePartialSources.has(healthSourceFor(entity))) {
        candidates.set(key, staleEntity(entity))
      }
    }

    for (const [key, entity] of candidates) {
      const source = healthSourceFor(entity)

      if (result.staleUnlistedSourcePrefixes?.some(prefix => source.startsWith(prefix))) {
        candidates.set(key, staleEntity(entity))
      }
    }

    for (const source of result.replacementSources ?? []) {
      for (const [key, entity] of candidates) {
        if (healthSourceFor(entity) === source) {
          candidates.delete(key)
        }
      }
    }

    for (const entity of result.entities) {
      candidates.set(entity.key, entity)
    }

    const sources =
      result.authoritative === false ? this.mergePartialSources(previous.sources, result.sources) : result.sources

    this.publishCandidate([...candidates.values()], sortedSources(sources))
  }

  private publishStale(): void {
    const previous = $lunarCitySnapshot.get()

    this.publishCandidate(
      [...previous.entities.values()].map(staleEntity),
      previous.sources.map(source => ({ ...source, authority: 'stale' as const }))
    )
  }

  private mergePartialSources(
    previous: readonly SourceHealth[],
    incoming: readonly SourceHealth[]
  ): readonly SourceHealth[] {
    const merged = new Map(previous.map(source => [source.source, source]))

    for (const source of incoming) {
      merged.set(source.source, source)
    }

    return [...merged.values()]
  }

  private publishCandidate(candidates: readonly LunarEntity[], sources: readonly SourceHealth[]): void {
    const previous = $lunarCitySnapshot.get()
    const next = new Map(candidates.map(entity => [entity.key, entity]))
    const removals = [...previous.entities.keys()].filter(key => !next.has(key))

    const upserts = [...next.values()].filter(entity => {
      const prior = previous.entities.get(entity.key)

      return prior === undefined || !sameEntity(prior, entity)
    })

    const sourcesChanged = !sameSources(previous.sources, sources)

    if (removals.length === 0 && upserts.length === 0 && !sourcesChanged) {
      this.scheduleFreshness(sources)

      return
    }

    const observedAt = Math.max(
      this.options.now?.() ?? Date.now(),
      ...sources.map(source => source.observedAt),
      ...candidates.map(entity => entity.observedAt)
    )

    const delta = freezeLunarDelta({
      observedAt,
      removals,
      revision: previous.revision + 1,
      sources,
      upserts
    })

    this.options.publish?.(delta) ?? applyLunarDelta(delta)
    this.scheduleFreshness(sources)
  }

  private scheduleFreshness(sources: readonly SourceHealth[]): void {
    if (this.freshnessTimer !== undefined) {
      clearTimeout(this.freshnessTimer)
      this.freshnessTimer = undefined
    }

    const freshnessMs = this.options.freshnessMs ?? 60_000

    const deadline = Math.min(
      ...sources.filter(source => source.authority === 'authoritative').map(source => source.observedAt + freshnessMs)
    )

    if (!Number.isFinite(deadline) || !this.active) {
      return
    }

    const delay = Math.max(0, deadline - (this.options.now?.() ?? Date.now()))
    this.freshnessTimer = setTimeout(() => {
      this.freshnessTimer = undefined

      if (this.active) {
        this.publishExpiredSources()
      }
    }, delay)
  }

  private publishExpiredSources(): void {
    const previous = $lunarCitySnapshot.get()
    const now = this.options.now?.() ?? Date.now()
    const freshnessMs = this.options.freshnessMs ?? 60_000

    const expired = new Set(
      previous.sources
        .filter(source => source.authority === 'authoritative' && source.observedAt + freshnessMs <= now)
        .map(source => source.source)
    )

    if (expired.size === 0) {
      this.scheduleFreshness(previous.sources)

      return
    }

    this.publishCandidate(
      [...previous.entities.values()].map(entity =>
        expired.has(healthSourceFor(entity)) ? staleEntity(entity) : entity
      ),
      previous.sources.map(source => (expired.has(source.source) ? { ...source, authority: 'stale' as const } : source))
    )
  }
}

export function createLunarCityReconciler(options: LunarCityReconcilerOptions): LunarCityReconciler {
  return new LunarCityReconciler(options)
}

function defaultLiveSources(): LunarCityLiveSources {
  return {
    $fleetRoster,
    $sessions,
    $subagentsBySession,
    legacyConnectionId: () => $connection.get()?.connectionId,
    legacySingleBackend: () => !hasRegistryTopology() && $connection.get()?.connectionId === 'local',
    readProfileRoster: (connectionId, profile) => requestGatewayForAgent(connectionId, profile, 'profiles.list', {}),
    readDelegationStatus: (connectionId, profile) =>
      requestGatewayForAgent(connectionId, profile, 'delegation.status', {}),
    readSessionList: (connectionId, profile) =>
      requestGatewayForAgent(connectionId, profile, 'session.list', { include_hidden: true, limit: 200 }),
    onEvent: listener => onGatewayEvent('*', listener as never),
    onFocus: listener => {
      const onVisibility = () => {
        if (document.visibilityState === 'visible') {
          listener()
        }
      }

      window.addEventListener('focus', listener)
      document.addEventListener('visibilitychange', onVisibility)

      return () => {
        window.removeEventListener('focus', listener)
        document.removeEventListener('visibilitychange', onVisibility)
      }
    },
    onRegistryChange: listener => {
      const offAtom = $connectionsRegistry.listen(listener)
      const offBridge = window.hermesDesktop?.connections?.onChanged?.(listener)

      return () => {
        offAtom()
        offBridge?.()
      }
    },
    refreshFleet: refreshFleetRoster,
    sessionRows: ownerLookupSessionRows
  }
}

function record(value: unknown): Record<string, unknown> {
  return value && typeof value === 'object' ? (value as Record<string, unknown>) : {}
}

function optionalString(value: unknown): string | undefined {
  return typeof value === 'string' && value.trim() ? value.trim() : undefined
}

function eventCursor(event: unknown): { scope: SequenceScope; sequence?: number; source?: SequenceSource } {
  const value = record(event)
  const payload = record(value.payload)
  const connectionId = optionalString(value.connectionId)
  const profile = optionalString(value.profile)
  const sessionId = optionalString(value.session_id)
  const rawSequence = value.seq ?? payload.seq
  const sequence = typeof rawSequence === 'number' && Number.isFinite(rawSequence) ? rawSequence : undefined

  // A backend sequence has meaning only alongside every owner component it
  // scopes.  An unscoped event remains a reread hint, never a shared cursor.
  const scope = { ...(connectionId ? { connectionId } : {}), ...(profile ? { profile } : {}) }

  if (!connectionId || !profile || !sessionId || sequence === undefined) {
    return { scope }
  }

  return { scope, sequence, source: { connectionId, profile, sessionId } }
}

/**
 * Starts the sole live writer for Lunar City.  Native events are hints; all
 * displayed data still comes from the bounded store reread.  The returned
 * disposer stops reconciliation before releasing every subscription.
 */
export function startLunarCityReconciler(options: StartLunarCityReconcilerOptions = {}): () => void {
  const sources = options.sources ?? defaultLiveSources()
  const optionalSources = options.optionalSources ?? sources.optionalSources ?? []
  const now = options.now ?? Date.now
  const sessionRows = sources.sessionRows ?? (() => sources.$sessions?.get() ?? [])
  let fleetReference = sources.$fleetRoster.get()
  let hasObservedBoundedRoster = Boolean(fleetReference && fleetReference.agents.length <= 256)
  let fleetObservedAt = 0
  let forceFleetRefresh = true
  let requestedBotMetadataGeneration = 1
  let appliedBotMetadataGeneration = 0
  let activeBotMetadataGeneration = 0
  let refreshingFleet = false
  let disposed = false
  let sessionSourceGeneration = 0
  const sessionOwnerObservations = new Map<string, Omit<SessionSourceObservation, 'fresh'>>()

  const sessionOwnerCache = new Map<
    string,
    { connectionId: string; observedAt: number; profile: string; rows: SessionInfo[] }
  >()

  const delegationOwnerCache = new Map<string, { batches: readonly OwnedSubagentRows[]; observedAt: number }>()
  const dirtySessionOwners = new Set<string>()
  const dirtyDelegationOwners = new Set<string>()
  const failedSessionOwners = new Set<string>()
  const failedDelegationOwners = new Set<string>()
  const registeredSessionOwners = new Map<string, { connectionId: string; profile: string }>()
  const retiredSessionOwners = new Map<string, { connectionId: string; profile: string }>()
  const botMetadataCache = new Map<string, { observedAt: number; payload: unknown; representativeProfile: string }>()
  const botMetadataProvenance = new Map<string, LunarPresentationMetadata>()
  const botRosterPlacementState = createBotRosterPlacementState()

  let enrichedFleetCache:
    { base: readonly LunarEntity[]; entities: readonly LunarEntity[]; roster: DesktopAgentRoster } | undefined

  const sameFleetBase = (left: readonly LunarEntity[], right: readonly LunarEntity[]): boolean => {
    if (left.length !== right.length) {
      return false
    }

    return left.every((entity, index) => {
      const candidate = right[index]

      return (
        candidate?.key === entity.key &&
        candidate.authority === entity.authority &&
        candidate.animation === entity.animation &&
        candidate.destination === entity.destination &&
        candidate.observedAt === entity.observedAt
      )
    })
  }

  const sessionConnectionReachable = (connectionId: string): boolean => {
    const roster = sources.$fleetRoster.get()
    const source = roster?.sources.find(candidate => candidate.connectionId === connectionId)

    return Boolean(roster && source?.reachable && !source.error)
  }

  const ownerKey = ownerObservationKey

  const runBounded = async <T>(jobs: readonly (() => Promise<T>)[], concurrency = 4): Promise<T[]> => {
    const results = new Array<T>(jobs.length)
    let next = 0

    const worker = async (): Promise<void> => {
      while (next < jobs.length) {
        const index = next
        next += 1
        results[index] = await jobs[index]!()
      }
    }

    await Promise.all(Array.from({ length: Math.min(concurrency, jobs.length) }, () => worker()))

    return results
  }

  const subagentProgress = (value: unknown): SubagentProgress | undefined => {
    const row = record(value)
    const id = optionalString(row.subagent_id)
    const rawStatus = optionalString(row.status)

    if (!id || !rawStatus) {
      return undefined
    }

    const status: SubagentProgress['status'] =
      rawStatus === 'completed' || rawStatus === 'failed' || rawStatus === 'interrupted' || rawStatus === 'queued'
        ? rawStatus
        : 'running'

    return {
      filesRead: [],
      filesWritten: [],
      goal: optionalString(row.goal) ?? '',
      id,
      parentId: null,
      startedAt: typeof row.started_at === 'number' && Number.isFinite(row.started_at) ? row.started_at : 0,
      status,
      stream: [],
      taskCount: 1,
      taskIndex: 0,
      toolCount: typeof row.tool_count === 'number' && Number.isFinite(row.tool_count) ? row.tool_count : 0,
      updatedAt: sessionNowForSubagent(row)
    }
  }

  const sessionNowForSubagent = (row: Record<string, unknown>): number =>
    typeof row.updated_at === 'number' && Number.isFinite(row.updated_at)
      ? row.updated_at
      : typeof row.started_at === 'number' && Number.isFinite(row.started_at)
        ? row.started_at
        : 0

  const readBotMetadata: ScopedProfilesListRequest = async (connectionId, representativeProfile) => {
    const cached = botMetadataCache.get(connectionId)

    if (appliedBotMetadataGeneration >= activeBotMetadataGeneration && cached) {
      return cached.payload
    }

    try {
      const payload = await sources.readProfileRoster!(connectionId, representativeProfile)
      const observedAt = now()
      botMetadataCache.set(connectionId, { observedAt, payload, representativeProfile })
      botMetadataProvenance.set(connectionId, {
        observedAt,
        source: `profiles:${connectionId}`,
        state: 'fresh'
      })

      return payload
    } catch (error) {
      if (cached) {
        botMetadataProvenance.set(connectionId, {
          observedAt: cached.observedAt,
          source: `profiles:${connectionId}`,
          state: 'stale'
        })

        return cached.payload
      }

      botMetadataProvenance.set(connectionId, { source: `profiles:${connectionId}`, state: 'unavailable' })

      throw error
    }
  }

  const reconciler = createLunarCityReconciler({
    freshnessMs: options.freshnessMs,
    now,
    read: async () => {
      const force = forceFleetRefresh
      forceFleetRefresh = false
      let fleetReadFailed = false
      let fleetError: string | undefined

      try {
        refreshingFleet = true
        const result = await sources.refreshFleet({ force })
        fleetReadFailed = result?.status === 'failed' || result?.status === 'partial'
        fleetError = result?.error

        if (result?.status === 'refreshed' && result.observedAt !== undefined) {
          fleetObservedAt = result.observedAt
        }
      } catch {
        // The existing roster remains valuable evidence, but it must retain
        // its last successful observation timestamp and become stale below.
        fleetReadFailed = true
        fleetError = 'Fleet refresh failed'
      } finally {
        refreshingFleet = false
      }

      const roster = sources.$fleetRoster.get()
      const rosterOversized = Boolean(roster && roster.agents.length > 256)

      const previousFleetSourceObservedAt = new Map(
        $lunarCitySnapshot
          .get()
          .sources.filter(source => source.source.startsWith('fleet:'))
          .map(source => [source.source, source.observedAt])
      )

      const fleetReadPartial =
        roster === null ||
        fleetReadFailed ||
        rosterOversized ||
        Boolean(roster.sources.some(source => !source.reachable || Boolean(source.error)))

      const fleetSourceObservedAt = new Map(
        roster?.sources.map(source => [
          source.connectionId,
          source.reachable && !source.error
            ? fleetObservedAt
            : (previousFleetSourceObservedAt.get(`fleet:${source.connectionId}`) ?? 0)
        ])
      )

      const normalizedFleet =
        roster && !rosterOversized
          ? normalizeRoster(roster, {
              fresh: !fleetReadFailed && fleetObservedAt > 0,
              observedAt: fleetObservedAt,
              sourceObservedAt: fleetSourceObservedAt
            })
          : rosterOversized || (roster === null && hasObservedBoundedRoster)
            ? {
                entities: [...$lunarCitySnapshot.get().entities.values()]
                  .filter(entity => entity.identity.kind === 'profile')
                  .map(staleEntity),
                sources: $lunarCitySnapshot
                  .get()
                  .sources.filter(source => source.source.startsWith('fleet:'))
                  .map(source => ({ ...source, authority: 'stale' as const }))
              }
            : { entities: [], sources: [] as readonly SourceHealth[] }

      let fleet = normalizedFleet
      const botMetadataGeneration = requestedBotMetadataGeneration

      if (roster && !rosterOversized && sources.readProfileRoster) {
        if (
          appliedBotMetadataGeneration >= botMetadataGeneration &&
          enrichedFleetCache?.roster === roster &&
          sameFleetBase(enrichedFleetCache.base, normalizedFleet.entities)
        ) {
          fleet = { ...normalizedFleet, entities: enrichedFleetCache.entities }
        } else {
          activeBotMetadataGeneration = botMetadataGeneration

          const entities = await enrichBotRosterEntities(
            roster,
            normalizedFleet.entities,
            readBotMetadata,
            botMetadataProvenance,
            { placementState: botRosterPlacementState }
          )

          fleet = { ...normalizedFleet, entities }
          enrichedFleetCache = { base: normalizedFleet.entities, entities, roster }
          appliedBotMetadataGeneration = Math.max(appliedBotMetadataGeneration, botMetadataGeneration)
          activeBotMetadataGeneration = 0
        }

        const currentSources = new Set(roster.sources.map(source => source.connectionId))

        for (const connectionId of botMetadataCache.keys()) {
          if (!currentSources.has(connectionId)) {
            botMetadataCache.delete(connectionId)
            botMetadataProvenance.delete(connectionId)
          }
        }
      }

      const sessionNow = now()
      const freshnessMs = options.freshnessMs ?? 60_000
      const priorSources = new Map($lunarCitySnapshot.get().sources.map(source => [source.source, source]))

      const reachableConnections = new Map(
        roster?.sources.map(source => [source.connectionId, source.reachable && !source.error]) ?? []
      )

      const cachedSessionRows = sessionRows()
      const cachedOwners = new Map<string, { connectionId: string; profile: string; rows: SessionInfo[] }>()

      for (const row of cachedSessionRows) {
        const owner = sessionOwnerRouteFromRow(row)

        if (!owner) {
          continue
        }

        const key = ownerKey(owner.connectionId, owner.profile)
        const batch = cachedOwners.get(key) ?? { ...owner, rows: [] }
        batch.rows.push(row)
        cachedOwners.set(key, batch)
      }

      const rosterOwners = new Map<string, { connectionId: string; profile: string; rows: SessionInfo[] }>()

      for (const agent of rosterOversized ? [] : (roster?.agents ?? [])) {
        const connectionId = agent.connectionId.trim()
        const profile = agent.profile.trim()

        if (!connectionId || !profile) {
          continue
        }

        const key = ownerKey(connectionId, profile)
        rosterOwners.set(key, { connectionId, profile, rows: cachedOwners.get(key)?.rows ?? [] })
      }

      const registeredOwners =
        roster === null
          ? hasObservedBoundedRoster
            ? new Map(
                [...registeredSessionOwners].map(([key, owner]) => [
                  key,
                  { ...owner, rows: sessionOwnerCache.get(key)?.rows ?? [] }
                ])
              )
            : new Map(cachedOwners)
          : rosterOversized
            ? new Map(
                [...registeredSessionOwners].map(([key, owner]) => [
                  key,
                  { ...owner, rows: sessionOwnerCache.get(key)?.rows ?? cachedOwners.get(key)?.rows ?? [] }
                ])
              )
            : rosterOwners

      if (roster !== null && !rosterOversized) {
        hasObservedBoundedRoster = true

        for (const [key, owner] of registeredSessionOwners) {
          if (!rosterOwners.has(key)) {
            retiredSessionOwners.set(key, owner)
            sessionOwnerCache.delete(key)
            sessionOwnerObservations.delete(key)
            delegationOwnerCache.delete(key)
            dirtySessionOwners.delete(key)
            dirtyDelegationOwners.delete(key)
            failedSessionOwners.delete(key)
            failedDelegationOwners.delete(key)

            while (retiredSessionOwners.size > MAX_RETIRED_SESSION_OWNERS) {
              retiredSessionOwners.delete(retiredSessionOwners.keys().next().value!)
            }
          }
        }

        for (const [key, owner] of rosterOwners) {
          retiredSessionOwners.delete(key)
          registeredSessionOwners.set(key, owner)
        }

        for (const key of registeredSessionOwners.keys()) {
          if (!rosterOwners.has(key)) {
            registeredSessionOwners.delete(key)
          }
        }
      }

      const ownerSetBounded = !rosterOversized && registeredOwners.size <= 256

      for (const key of registeredOwners.keys()) {
        if (!sessionOwnerCache.has(key)) {
          dirtySessionOwners.add(key)
          dirtyDelegationOwners.add(key)
        }
      }

      if (sources.readSessionList && ownerSetBounded && roster !== null) {
        const jobs = [...registeredOwners].flatMap(([key, owner]) => {
          if (
            (!dirtySessionOwners.has(key) && !dirtyDelegationOwners.has(key)) ||
            !sessionConnectionReachable(owner.connectionId)
          ) {
            return []
          }

          return [
            async () => {
              let rows = sessionOwnerCache.get(key)?.rows

              if (dirtySessionOwners.has(key)) {
                try {
                  const payload = record(await sources.readSessionList!(owner.connectionId, owner.profile))

                  const rawRows =
                    Array.isArray(payload.sessions) && payload.sessions.length <= 200 ? payload.sessions : null

                  if (!rawRows) {
                    throw new Error('malformed session list')
                  }

                  rows = rawRows.flatMap(row => {
                    const id = optionalString(record(row).id)

                    return id
                      ? [
                          {
                            ...(row as SessionInfo),
                            connection_id: owner.connectionId,
                            id,
                            profile: owner.profile
                          }
                        ]
                      : []
                  })

                  if (rows.length !== rawRows.length || new Set(rows.map(row => row.id)).size !== rows.length) {
                    throw new Error('malformed session identities')
                  }

                  sessionSourceGeneration += 1
                  sessionOwnerCache.set(key, { ...owner, observedAt: sessionNow, rows })
                  sessionOwnerObservations.set(key, { generation: sessionSourceGeneration, observedAt: sessionNow })
                  failedSessionOwners.delete(key)

                  const currentParents = new Set(rows.map(row => row.id))
                  const cachedDelegation = delegationOwnerCache.get(key)

                  if (cachedDelegation) {
                    delegationOwnerCache.set(key, {
                      ...cachedDelegation,
                      batches: cachedDelegation.batches.filter(batch => currentParents.has(batch.owner.sessionId))
                    })
                  }
                } catch {
                  failedSessionOwners.add(key)
                  failedDelegationOwners.add(key)
                  dirtyDelegationOwners.delete(key)

                  return
                } finally {
                  dirtySessionOwners.delete(key)
                }
              }

              if (sources.readDelegationStatus && dirtyDelegationOwners.has(key) && rows) {
                try {
                  const delegation = record(await sources.readDelegationStatus(owner.connectionId, owner.profile))

                  const active =
                    Array.isArray(delegation.active) && delegation.active.length <= 512 ? delegation.active : null

                  if (!active) {
                    throw new Error('malformed delegation status')
                  }

                  const sessionsById = new Set(rows.map(row => row.id))
                  const batches = new Map<string, OwnedSubagentRows>()

                  for (const value of active) {
                    const raw = record(value)
                    const parentSessionId = optionalString(raw.owner_agent_session_id)
                    const progress = subagentProgress(raw)

                    if (!parentSessionId || !progress || !sessionsById.has(parentSessionId)) {
                      continue
                    }

                    const ownerIdentity = {
                      connectionId: owner.connectionId,
                      kind: 'session' as const,
                      profile: owner.profile,
                      sessionId: parentSessionId
                    }

                    const batch = batches.get(parentSessionId) ?? { owner: ownerIdentity, rows: [] }
                    batches.set(parentSessionId, {
                      owner: batch.owner,
                      rows: [...batch.rows, { ...progress, parentId: parentSessionId }]
                    })
                  }

                  delegationOwnerCache.set(key, { batches: [...batches.values()], observedAt: sessionNow })
                  failedDelegationOwners.delete(key)
                } catch {
                  failedDelegationOwners.add(key)
                } finally {
                  dirtyDelegationOwners.delete(key)
                }
              }
            }
          ]
        })

        await runBounded(jobs)
      }

      const effectiveSessionRows = [...registeredOwners].flatMap(([key, owner]) => {
        const cached = sessionOwnerCache.get(key)

        return cached?.rows ?? owner.rows
      })

      const currentSessionObservations = new Map<string, SessionSourceObservation>()
      const currentDelegationObservations = new Map<string, SessionSourceObservation>()
      const sessionHealth: SourceHealth[] = []
      const delegationHealth: SourceHealth[] = []
      const authoritativeSessionSources: string[] = []
      const authoritativeDelegationSources: string[] = []
      const parentFilteredDelegationSources: string[] = []

      for (const [key, owner] of registeredOwners) {
        const reachable = reachableConnections.get(owner.connectionId) === true
        const sessionObserved = sessionOwnerObservations.get(key)
        const priorSession = priorSources.get(sessionSourceName(owner.connectionId, owner.profile))
        const sessionObservedAt = sessionObserved?.observedAt ?? priorSession?.observedAt ?? 0

        const sessionFresh = Boolean(
          ownerSetBounded &&
          reachable &&
          sessionObserved &&
          !failedSessionOwners.has(key) &&
          sessionObservedAt + freshnessMs > sessionNow
        )

        currentSessionObservations.set(key, {
          fresh: sessionFresh,
          generation: sessionObserved?.generation ?? 0,
          observedAt: sessionObservedAt
        })
        const sessionName = sessionSourceName(owner.connectionId, owner.profile)
        sessionHealth.push({
          authority: sessionFresh ? 'authoritative' : sessionObservedAt > 0 ? 'stale' : 'unknown',
          ...(failedSessionOwners.has(key) ? { error: 'Session refresh failed' } : {}),
          observedAt: sessionObservedAt,
          source: sessionName
        })

        if (sessionFresh) {
          authoritativeSessionSources.push(sessionName)
        }

        const delegationCached = delegationOwnerCache.get(key)
        const priorDelegation = priorSources.get(delegationSourceName(owner.connectionId, owner.profile))
        const delegationObservedAt = delegationCached?.observedAt ?? priorDelegation?.observedAt ?? 0

        const delegationFresh = Boolean(
          ownerSetBounded &&
          reachable &&
          delegationCached &&
          !failedDelegationOwners.has(key) &&
          delegationObservedAt + freshnessMs > sessionNow
        )

        currentDelegationObservations.set(key, {
          fresh: delegationFresh,
          generation: 0,
          observedAt: delegationObservedAt
        })

        if (sources.readDelegationStatus) {
          const delegationName = delegationSourceName(owner.connectionId, owner.profile)
          delegationHealth.push({
            authority: delegationFresh ? 'authoritative' : delegationObservedAt > 0 ? 'stale' : 'unknown',
            ...(failedDelegationOwners.has(key) ? { error: 'Delegation refresh failed' } : {}),
            observedAt: delegationObservedAt,
            source: delegationName
          })

          if (delegationFresh) {
            authoritativeDelegationSources.push(delegationName)
          } else if (sessionFresh && delegationCached) {
            parentFilteredDelegationSources.push(delegationName)
          }
        }
      }

      const sessions = normalizeSessions(effectiveSessionRows, {
        fresh: false,
        legacyConnectionId: sources.legacyConnectionId?.(),
        legacySingleBackend: sources.legacySingleBackend(),
        observedAt: 0,
        sourceObservations: currentSessionObservations
      })

      const legacySubagents = normalizeSubagents(sources.$subagentsBySession.get(), sessions.entities, {
        fresh: false,
        observedAt: 0,
        sourceObservations: sources.readDelegationStatus ? currentDelegationObservations : currentSessionObservations
      })

      const ownedSubagents = normalizeOwnedSubagents(
        [...delegationOwnerCache.values()].flatMap(entry => entry.batches),
        {
          fresh: false,
          observedAt: 0,
          sourceObservations: currentDelegationObservations
        }
      )

      const subagents = {
        entities: [
          ...new Map(
            [...legacySubagents.entities, ...ownedSubagents.entities].map(entity => [entity.key, entity])
          ).values()
        ]
      }

      const optionalReads: ReconcileReadResult[] = []

      for (const source of optionalSources) {
        try {
          optionalReads.push(await source.read())
        } catch {
          optionalReads.push({ authoritative: false, entities: [], sources: [] })
        }
      }

      const previousEntities = [...$lunarCitySnapshot.get().entities.values()]

      const optionalEntities = optionalReads.flatMap(read => {
        if (read.authoritative !== false) {
          return read.entities
        }

        const replacementSources = new Set(read.replacementSources ?? [])

        const retainedSources = new Set(
          read.sources.map(source => source.source).filter(source => !replacementSources.has(source))
        )

        return [
          ...previousEntities.filter(entity => retainedSources.has(healthSourceFor(entity))).map(staleEntity),
          ...read.entities
        ]
      })

      const optionalReplacementSources = optionalReads.flatMap(
        read =>
          read.replacementSources ?? (read.authoritative !== false ? read.sources.map(source => source.source) : [])
      )

      const authoritativeFleetSources = fleet.sources
        .filter(source => source.authority === 'authoritative')
        .map(source => source.source)

      const retiredSessionSources = [...retiredSessionOwners.values()].flatMap(owner => {
        const names = [
          sessionSourceName(owner.connectionId, owner.profile),
          ...(sources.readDelegationStatus ? [delegationSourceName(owner.connectionId, owner.profile)] : [])
        ]

        return names.flatMap(source => {
          const prior = priorSources.get(source)

          return prior ? [{ ...prior, authority: 'stale' as const, error: 'Registered session owner removed' }] : []
        })
      })

      const sessionReadPartial =
        Boolean(sources.readSessionList) &&
        (!ownerSetBounded ||
          retiredSessionOwners.size > 0 ||
          authoritativeSessionSources.length !== registeredOwners.size ||
          (Boolean(sources.readDelegationStatus) && authoritativeDelegationSources.length !== registeredOwners.size))

      retiredSessionOwners.clear()

      const registryHealth: SourceHealth[] = rosterOversized
        ? [
            {
              authority: 'partial',
              error: 'Registered session owner limit exceeded',
              observedAt: sessionNow,
              source: 'session-registry:overflow'
            }
          ]
        : roster === null
          ? [
              {
                authority: 'partial',
                error: 'Registered session roster unavailable',
                observedAt: sessionNow,
                source: 'session-registry:unavailable'
              }
            ]
          : []

      return {
        authoritative:
          !fleetReadPartial && !sessionReadPartial && optionalReads.every(read => read.authoritative !== false),
        entities: [...fleet.entities, ...sessions.entities, ...subagents.entities, ...optionalEntities],
        replacementSources: [
          ...authoritativeFleetSources,
          ...authoritativeSessionSources,
          ...authoritativeDelegationSources,
          ...parentFilteredDelegationSources,
          ...optionalReplacementSources
        ],
        sources: [
          ...fleet.sources.map(source =>
            fleetReadFailed ? { ...source, error: fleetError ?? 'Fleet refresh failed' } : source
          ),
          ...(sources.readSessionList ? sessionHealth : sessions.sources),
          ...delegationHealth,
          ...retiredSessionSources,
          ...registryHealth,
          ...optionalReads.flatMap(read => [...read.sources])
        ],
        staleUnlistedSourcePrefixes: [
          ...new Set([
            ...(roster === null && hasObservedBoundedRoster ? ['fleet:', 'session:', 'delegation:'] : []),
            ...optionalReads.flatMap(read => read.staleUnlistedSourcePrefixes ?? [])
          ])
        ]
      }
    }
  })

  const disposers: (() => void)[] = []
  disposers.push(
    sources.$fleetRoster.listen(roster => {
      if (roster !== fleetReference) {
        fleetReference = roster
        fleetObservedAt = now()
        requestedBotMetadataGeneration += 1

        for (const key of sessionOwnerCache.keys()) {
          dirtySessionOwners.add(key)
          dirtyDelegationOwners.add(key)
        }

        if (!refreshingFleet) {
          reconciler.invalidate('fleet')
        }
      }
    })
  )
  disposers.push(sources.$subagentsBySession.listen(() => reconciler.invalidate('subagents')))

  if (sources.onFocus) {
    disposers.push(
      sources.onFocus(() => {
        for (const key of sessionOwnerCache.keys()) {
          dirtySessionOwners.add(key)
          dirtyDelegationOwners.add(key)
        }

        requestedBotMetadataGeneration += 1
        reconciler.invalidate('focus')
      })
    )
  }

  if (sources.onRegistryChange) {
    disposers.push(
      sources.onRegistryChange(() => {
        for (const key of sessionOwnerCache.keys()) {
          dirtySessionOwners.add(key)
          dirtyDelegationOwners.add(key)
        }

        forceFleetRefresh = true
        requestedBotMetadataGeneration += 1
        reconciler.invalidate('registry')
      })
    )
  }

  if (sources.onEvent) {
    disposers.push(
      sources.onEvent(event => {
        const type = optionalString(record(event).type)
        const cursor = eventCursor(event)
        const connectionId = optionalString(record(event).connectionId)
        const profile = optionalString(record(event).profile)

        if (connectionId && profile) {
          const key = ownerKey(connectionId, profile)
          dirtySessionOwners.add(key)
          dirtyDelegationOwners.add(key)
        }

        if (type === 'gateway.ready') {
          forceFleetRefresh = true
          requestedBotMetadataGeneration += 1
          reconciler.resetSequences(cursor.scope)
        } else if (type === 'profile.changed' || type === 'profiles.changed' || type === 'profile.ui_meta.changed') {
          requestedBotMetadataGeneration += 1
        }

        reconciler.acceptEvent(cursor)
      })
    )
  }

  for (const source of optionalSources) {
    if (source.start) {
      disposers.push(source.start(() => reconciler.invalidate('optional-source')))
    }
  }

  const stop = reconciler.start()

  return () => {
    if (disposed) {
      return
    }

    disposed = true
    stop()

    for (const dispose of disposers.splice(0)) {
      dispose()
    }
  }
}
