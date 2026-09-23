import crypto, { randomUUID } from 'node:crypto'

import type {
  ManagedRolloutIpcAdapter,
  ManagedRolloutIpcCapabilities,
  ManagedRolloutIpcCommand
} from './managed-rollout-ipc'
import {
  createManagedRolloutCoordinator,
  createManagedRolloutState,
  type ManagedRolloutAuthorization,
  type ManagedRolloutCoordinatorDependencies,
  type ManagedRolloutEvidenceAdapter,
  type ManagedRolloutPhase as CoordinatorPhase,
  type ManagedRolloutState,
  type ManagedRolloutTarget
} from './managed-rollout-coordinator'
import {
  verifyApplicableAssurance,
  verifyReviewedGitSource,
  type TrustedAssuranceReader,
  type TrustedSourceReader,
  type VerifiedAssuranceEvidence
} from './managed-rollout-assurance'
import {
  verifyTrustedInventory,
  type InventoryObservation,
  type TrustedInventoryReader,
  type TrustedInventorySnapshot,
  type VerifiedInventoryRow
} from './managed-rollout-inventory'
import {
  createPreflightReview,
  canonicalPlanDigest,
  type TargetResolution,
  validateTargetResolution,
  ReviewTokenStore
} from './managed-rollout-preflight'
import {
  type JournalAck,
  type JournalEvidenceFact,
  type JournalEventInput,
  type JournalRecord,
  type JournalSnapshot,
  type ManagedRolloutJournal,
  type UnresolvedFenceChange
} from './managed-rollout-journal'
import type {
  ManagedConnectionUpdateResult,
  ManagedSshUpdateIntent
} from './managed-ssh-update'
import type {
  ManagedSshLaunchCapability,
  ManagedSshUpdateService
} from './managed-ssh-update-service'
import {
  validateRolloutCommand,
  validateRolloutPlan,
  validateRolloutSnapshot,
  type HealthEvidence,
  type PlanChange,
  type RolloutAction,
  type RolloutCommand,
  type RolloutDraft,
  type RolloutPhase,
  type RolloutPlan,
  type RolloutSnapshot,
  type TargetAttempt,
  type TargetPhase,
  type LaunchState,
  type ReviewedSourceBinding,
  type PromotionPolicy
} from '../src/lib/managed-rollout-contract'

export const MANAGED_ROLLOUT_UNAVAILABLE_REASON = 'trusted-rollout-dependencies-unavailable'
export const MAX_PROVIDER_PAGE_SIZE = 50

const TERMINAL_PHASES = new Set<RolloutPhase>([
  'stopped',
  'completed',
  'completed-with-exclusions'
])
const SUCCESSFUL_OUTCOMES = new Set(['updated', 'already-current'])
const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/

export interface ManagedRolloutTargetResolution {
  plan: RolloutPlan
  resolution: TargetResolution
}

export interface ManagedRolloutObservation {
  outcome: 'updated' | 'already-current' | 'failed' | 'refused' | 'unverified'
  receipt: ManagedConnectionUpdateResult['receipt']
  health: HealthEvidence | null
  /** Optional echo used to detect an adapter that crossed the wrong transaction. */
  authorization?: ManagedRolloutAuthorization
}

export interface ManagedRolloutObservationReader {
  observe: (input: {
    authorization: ManagedRolloutAuthorization
    update: ManagedConnectionUpdateResult
  }) => Promise<ManagedRolloutObservation>
  reprobe?: (authorization: ManagedRolloutAuthorization) => Promise<{
    correlationId: string
    outcome: ManagedRolloutObservation['outcome']
    terminal: boolean
  }>
  recover?: (authorization: ManagedRolloutAuthorization) => Promise<{
    correlationId: string
    clearanceProved: boolean
  }>
}

export interface ManagedRolloutProviderDependencies {
  inventoryReader: TrustedInventoryReader
  sourceReader: TrustedSourceReader
  assuranceReader: TrustedAssuranceReader
  resolveTarget: (request: {
    connectionIds: string[]
    inventoryRevision: string
    retryOf: string | null
  }) => Promise<ManagedRolloutTargetResolution>
  journal: ManagedRolloutJournal
  managedSshUpdateService: Pick<ManagedSshUpdateService, 'issueLaunchCapability' | 'request'>
  observe: ManagedRolloutObservationReader
  evidence: ManagedRolloutEvidenceAdapter
  ready?: () => boolean
  now?: () => number
  nowMono?: () => number
  processGeneration?: number
}

export interface ManagedRolloutProvider extends ManagedRolloutIpcAdapter {
  inventory: () => Promise<unknown>
  resolveTarget: (request: {
    connectionIds: string[]
    inventoryRevision: string
    retryOf: string | null
  }) => Promise<unknown>
  preflight: (draft: unknown) => Promise<unknown>
  start: (request: { token: string; requestId: string }) => Promise<unknown>
  waitForIdle: () => Promise<void>
}

type Coordinator = ReturnType<typeof createManagedRolloutCoordinator>
type LaunchPromise = Promise<ManagedConnectionUpdateResult>

type ResolvedEntry = {
  plan: RolloutPlan
  resolution: TargetResolution
}

type ReviewSession = {
  token: string
  requestId: string
  rolloutId: string
  plan: RolloutPlan
  expiresAt: number
  planDigest: string
}

type ObservationState = {
  receipt: ManagedConnectionUpdateResult['receipt']
  health: HealthEvidence | null
}

type Deferred = {
  promise: Promise<void>
  resolve: () => void
}

type Runtime = {
  id: string
  plan: RolloutPlan
  targets: Map<string, ManagedRolloutTarget>
  authorizations: Map<string, ManagedRolloutAuthorization>
  observations: Map<string, ObservationState>
  launches: Map<string, LaunchPromise>
  coordinator: Coordinator
  snapshot: RolloutSnapshot
  queue: Promise<void>
  handoffBarrier: Deferred | null
  runPromise: Promise<void> | null
}

function unavailable(): Error {
  return new Error(MANAGED_ROLLOUT_UNAVAILABLE_REASON)
}

function isFunction(value: unknown): value is (...args: never[]) => unknown {
  return typeof value === 'function'
}

function isCompleteDependencies(value: Partial<ManagedRolloutProviderDependencies>): value is ManagedRolloutProviderDependencies {
  return Boolean(
    value.inventoryReader && isFunction(value.inventoryReader.capture) &&
    value.sourceReader && isFunction(value.sourceReader.git) && isFunction(value.sourceReader.nowMono) &&
    value.assuranceReader && isFunction(value.assuranceReader.readEvidence) && isFunction(value.assuranceReader.readProfile) &&
    isFunction(value.resolveTarget) && value.journal &&
    isFunction(value.journal.create) && isFunction(value.journal.record) &&
    isFunction(value.journal.read) && isFunction(value.journal.history) && isFunction(value.journal.events) &&
    value.managedSshUpdateService && isFunction(value.managedSshUpdateService.issueLaunchCapability) &&
    isFunction(value.managedSshUpdateService.request) &&
    value.observe && isFunction(value.observe.observe) &&
    value.evidence && isFunction(value.evidence.sweep)
  )
}

function clone<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T
}

function isoNow(now: () => number): string {
  const value = now()
  if (!Number.isSafeInteger(value)) throw new Error('rollout-clock-invalid')
  return new Date(value).toISOString()
}

function boundedText(value: unknown, fallback: string): string {
  if (typeof value !== 'string' || value.length === 0 || value.length > 500 || /[\x00-\x1f\x7f]/.test(value)) {
    return fallback
  }
  return value
}

function canonicalJson(value: unknown, stack = new Set<object>()): string {
  if (value === null) return 'null'
  if (typeof value === 'string' || typeof value === 'boolean') return JSON.stringify(value)
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) throw new Error('invalid-command-payload')
    return JSON.stringify(value)
  }
  if (Array.isArray(value)) return `[${value.map(item => canonicalJson(item, stack)).join(',')}]`
  if (typeof value !== 'object') throw new Error('invalid-command-payload')
  if (stack.has(value)) throw new Error('invalid-command-payload')
  stack.add(value)
  const result = `{${Object.keys(value).sort().map(key => `${JSON.stringify(key)}:${canonicalJson((value as Record<string, unknown>)[key], stack)}`).join(',')}}`
  stack.delete(value)
  return result
}

function payloadDigest(value: unknown): string {
  return crypto.createHash('sha256').update(canonicalJson(value), 'utf8').digest('hex')
}

function event(
  kind: string,
  installId: string | null = null,
  reason: string | null = null,
  actor: JournalEventInput['actor'] = 'system'
): JournalEventInput {
  return { kind, actor, installId, reason, evidenceDigest: null }
}

function fact(
  kind: JournalEvidenceFact['kind'],
  rolloutId: string,
  installId: string | null,
  correlationId: string | null,
  observedAt: string,
  basis: string
): JournalEvidenceFact {
  return { kind, rolloutId, installId, correlationId, observedAt, basis }
}

function fence(rolloutId: string, installId: string, correlationId: string, recordedAt: string) {
  return {
    key: `managed-rollout:${rolloutId}:${installId}:${correlationId}`,
    rolloutId,
    installId,
    correlationId,
    reason: 'remote-launch-settlement-required',
    recordedAt
  }
}

function createDeferred(): Deferred {
  let resolvePromise: (() => void) | null = null
  const promise = new Promise<void>(resolve => {
    resolvePromise = resolve
  })
  return {
    promise,
    resolve: () => resolvePromise?.()
  }
}

function capabilities(available: boolean): ManagedRolloutIpcCapabilities {
  return available
    ? { protocol: 1, available: true, reason: null, maxConcurrency: 4, maxInstallations: 500 }
    : { protocol: 1, available: false, reason: MANAGED_ROLLOUT_UNAVAILABLE_REASON, maxConcurrency: 0, maxInstallations: 0 }
}

function targetResolutionOutput(entry: ResolvedEntry): Record<string, unknown> {
  return {
    resolutionId: entry.resolution.id,
    fingerprint: entry.resolution.fingerprint,
    createdAt: entry.resolution.createdAt,
    expiresAt: entry.resolution.expiresAt,
    inventoryRevision: entry.plan.inventoryRevision ?? null,
    target: clone(entry.plan.target),
    plan: clone(entry.plan)
  }
}

function inventoryOutput(snapshot: TrustedInventorySnapshot): Record<string, unknown> {
  if (!snapshot || !Number.isFinite(snapshot.capturedMono) || snapshot.capturedMono < 0) throw new Error('inventory-unavailable')
  if (typeof snapshot.inventoryRevision !== 'string' || !snapshot.inventoryRevision || snapshot.inventoryRevision.length > 256) {
    throw new Error('inventory-revision-invalid')
  }
  if (!Array.isArray(snapshot.observations) || snapshot.observations.length > 500) throw new Error('inventory-bounded-limit')
  const observations = snapshot.observations.map((item: InventoryObservation) => ({
    installId: item.installId,
    connectionId: item.connectionId,
    aliasConnectionIds: [...item.aliasConnectionIds].slice(0, 500),
    codeRoot: item.codeRoot,
    repositoryId: item.repositoryId,
    headSha: item.headSha,
    requiredScopeIds: [...item.requiredScopeIds].slice(0, 500),
    source: {
      connectionId: item.source.connectionId,
      connectionConfigRevision: item.source.connectionConfigRevision,
      verifiedHostKeyFingerprint: item.source.verifiedHostKeyFingerprint,
      remoteUser: item.source.remoteUser,
      port: item.source.port,
      configuredProfile: item.source.configuredProfile,
      configuredCodePath: item.source.configuredCodePath
    }
  }))
  return {
    inventoryRevision: snapshot.inventoryRevision,
    capturedMono: snapshot.capturedMono,
    observations
  }
}

function policy(value: PromotionPolicy): 'manual' | 'auto-after-canary' {
  return value === 'auto-if-healthy' ? 'auto-after-canary' : 'manual'
}

function canonicalPhase(value: CoordinatorPhase): RolloutPhase {
  switch (value) {
    case 'queued':
    case 'running': return 'running'
    case 'awaiting-promotion': return 'awaiting-promotion'
    case 'paused': return 'paused'
    case 'stopped': return 'stopped'
    case 'attention-required': return 'attention-required'
    case 'reconciling': return 'reconciling'
    case 'completed': return 'completed'
    case 'completed-with-exclusions': return 'completed-with-exclusions'
  }
}

function targetPhase(state: ManagedRolloutState['attempts'][string]['state']): TargetPhase {
  switch (state) {
    case 'updated': return 'updated'
    case 'already-current': return 'already-current'
    case 'failed': return 'failed'
    case 'refused': return 'refused'
    case 'unverified': return 'unverified'
    case 'recovery-required': return 'recovery-required'
    case 'skipped':
    case 'cancelled-before-launch': return 'skipped'
    case 'authorized': return 'awaiting-receipt'
    case 'observed': return 'verifying'
    case 'intent-recorded': return 'draining'
    case 'none': return 'queued'
  }
}

function launchState(state: ManagedRolloutState['attempts'][string]['state']): LaunchState {
  switch (state) {
    case 'none':
    case 'skipped':
    case 'cancelled-before-launch': return 'none'
    case 'intent-recorded': return 'intent-recorded'
    case 'authorized': return 'authorized'
    case 'observed':
    case 'updated':
    case 'already-current':
    case 'failed':
    case 'refused': return 'observed'
    case 'unverified':
    case 'recovery-required': return 'authorized'
  }
}

function receiptOutput(
  receipt: ManagedConnectionUpdateResult['receipt'],
  targetSha: string,
  installId: string
): RolloutSnapshot['attempts'][number]['receipt'] {
  if (!receipt) return null
  return {
    correlationId: receipt.correlationId,
    installId,
    requestedSha: targetSha,
    preSha: receipt.preSha ?? null,
    postSha: receipt.postSha ?? null,
    outcome: boundedText(receipt.outcome, 'unknown'),
    startedAt: receipt.startedAt ?? null,
    finishedAt: receipt.finishedAt ?? null,
    stopReason: receipt.stopReason ?? null
  }
}

function snapshotFromState(runtime: Runtime, state: ManagedRolloutState, current: RolloutSnapshot): RolloutSnapshot {
  const attempts: TargetAttempt[] = [...runtime.targets.values()].map(target => {
    const attempt = state.attempts[target.installId]
    const row = runtime.plan.rows.find(item => item.installId === target.installId)
    const observation = runtime.observations.get(target.installId)
    const excluded = Boolean(attempt.excluded)
    const skipReason = excluded
      ? 'operator-excluded'
      : attempt.state === 'cancelled-before-launch'
        ? 'cancelled-before-launch'
        : attempt.state === 'skipped'
          ? 'stop-recorded'
          : null
    const reasons = attempt.reason ? [boundedText(attempt.reason, 'coordinator-refused')] : []
    return {
      identity: {
        connectionId: target.connectionId,
        installId: target.installId,
        aliasConnectionIds: [],
        label: target.installId,
        displayAddress: target.connectionId,
        installationFingerprint: target.installationFingerprint,
        sourceFingerprint: target.sourceFingerprint,
        admittedSha: row?.admittedHead as string
      },
      correlationId: target.correlationId,
      wave: target.wave,
      phase: targetPhase(attempt.state),
      launchState: launchState(attempt.state),
      requiredScopeIds: row?.requiredScopeIds ? [...row.requiredScopeIds] : null,
      skipReason,
      reprobes: attempt.reprobeCount,
      receipt: receiptOutput(observation?.receipt ?? null, target.targetSha, target.installId),
      health: observation?.health ? clone(observation.health) : null,
      recoveryRequired: attempt.state === 'unverified' || attempt.state === 'recovery-required',
      reasons
    }
  })
  const phase = canonicalPhase(state.phase)
  const finishedAt = TERMINAL_PHASES.has(phase) ? current.finishedAt ?? current.updatedAt : null
  return validateRolloutSnapshot({
    schemaVersion: 1,
    id: runtime.id,
    revision: current.revision,
    createdAt: current.createdAt,
    updatedAt: current.updatedAt,
    finishedAt,
    retryOf: runtime.plan.retryOf,
    archivedAt: current.archivedAt ?? null,
    target: clone(runtime.plan.target),
    phase,
    activeWave: state.currentWave,
    concurrency: runtime.plan.concurrency,
    promotionPolicy: runtime.plan.promotionPolicy,
    canaryApproved: state.canaryApproved,
    continuationRequired: state.continuationRequired,
    attempts,
    eventCount: current.eventCount
  })
}

function initialSnapshot(
  id: string,
  plan: RolloutPlan,
  at: string,
  correlations?: ReadonlyMap<string, string>
): RolloutSnapshot {
  const attempts = plan.rows.map(row => {
    if (!row.reviewedSource || !row.requiredScopeIds || !row.admittedHead) throw new Error('rollout-plan-provenance-incomplete')
    const wave = plan.waves.findIndex(values => values.includes(row.installId))
    return {
      identity: {
        connectionId: row.connectionId,
        installId: row.installId,
        aliasConnectionIds: [],
        label: row.installId,
        displayAddress: row.connectionId,
        installationFingerprint: row.installationFingerprint,
        sourceFingerprint: row.sourceFingerprint,
        admittedSha: row.admittedHead
      },
      correlationId: correlations?.get(row.installId) ?? randomUUID(),
      wave,
      phase: 'queued' as const,
      launchState: 'none' as const,
      requiredScopeIds: [...row.requiredScopeIds],
      skipReason: null,
      reprobes: 0,
      receipt: null,
      health: null,
      recoveryRequired: false,
      reasons: []
    }
  })
  return validateRolloutSnapshot({
    schemaVersion: 1,
    id,
    revision: 0,
    createdAt: at,
    updatedAt: at,
    finishedAt: null,
    retryOf: plan.retryOf,
    archivedAt: null,
    target: clone(plan.target),
    phase: 'running',
    activeWave: 0,
    concurrency: plan.concurrency,
    promotionPolicy: plan.promotionPolicy,
    canaryApproved: false,
    continuationRequired: false,
    attempts,
    eventCount: 0
  })
}

function normalizedCommand(value: ManagedRolloutIpcCommand): RolloutCommand {
  const input = value as unknown as Record<string, unknown>
  return validateRolloutCommand({
    id: input.id,
    expectedRevision: input.expectedRevision ?? input.revision,
    requestId: input.requestId,
    action: input.action ?? input.kind,
    installId: input.installId ?? null,
    reason: input.reason ?? null,
    promotionPolicy: input.promotionPolicy ?? null
  })
}

function commandEventKind(action: RolloutAction, accepted: boolean): string {
  if (!accepted) return 'operator-disposition'
  switch (action) {
    case 'promote': return 'promoted'
    case 'pause': return 'pause-requested'
    case 'resume': return 'resumed'
    case 'stop': return 'stop-requested'
    case 'reprobe': return 'reprobed'
    case 'recover': return 'recovered'
    case 'exclude': return 'attempt-excluded'
    case 'set-policy': return 'policy-changed'
    case 'archive': return 'archived'
  }
}

function staleAck(id: string, revision: number, code: string, message: string): Record<string, unknown> {
  return { ok: false, id, revision, code, message, changes: [] }
}

function activePhase(phase: string): boolean {
  return !TERMINAL_PHASES.has(phase as RolloutPhase)
}

function exactConnectionSet(plan: RolloutPlan, connectionIds: readonly string[]): boolean {
  const requested = new Set(connectionIds)
  const rows = plan.rows.map(row => row.connectionId)
  return rows.length === requested.size && new Set(rows).size === rows.length && rows.every(id => requested.has(id))
}

function sameAuthorization(left: ManagedRolloutAuthorization, right: ManagedRolloutAuthorization): boolean {
  return left.rolloutId === right.rolloutId && left.installId === right.installId && left.connectionId === right.connectionId &&
    left.installationFingerprint === right.installationFingerprint && left.sourceFingerprint === right.sourceFingerprint &&
    left.targetSha === right.targetSha && left.correlationId === right.correlationId &&
    left.queueGeneration === right.queueGeneration && JSON.stringify(left.reviewedSource) === JSON.stringify(right.reviewedSource)
}

function successfulHealth(
  health: HealthEvidence | null,
  authorization: ManagedRolloutAuthorization,
  requiredScopeIds: readonly string[]
): boolean {
  if (!health || health.installId !== authorization.installId || health.checkoutSha !== authorization.targetSha) return false
  if (!health.installReady || !health.markerClear || !health.receiptCorrelated || !health.receiptSucceeded ||
      !health.dependencyReady || !health.recoveryClear || health.scopeCapture !== 'complete') return false
  const expected = new Set(requiredScopeIds)
  const actual = new Set<string>()
  for (const scope of health.scopes) {
    actual.add(scope.scopeId)
    if (!scope.restored || !scope.ready || scope.codeSha !== authorization.targetSha || !scope.processIdentityVerified) return false
  }
  return expected.size === actual.size && [...expected].every(scopeId => actual.has(scopeId))
}

function providerUnavailable(): ManagedRolloutProvider {
  const reject = async (): Promise<never> => { throw unavailable() }
  return {
    capabilities: async () => capabilities(false),
    inventory: reject,
    resolveTarget: reject,
    preflight: reject,
    start: reject,
    activeRevision: async () => null,
    read: reject,
    get: reject,
    command: reject,
    history: reject,
    events: reject,
    waitForIdle: async () => undefined
  }
}

export function createManagedRolloutProvider(
  partial: Partial<ManagedRolloutProviderDependencies>
): ManagedRolloutProvider {
  if (!isCompleteDependencies(partial)) return providerUnavailable()
  const deps = partial
  const reviewTokens = new ReviewTokenStore()
  const resolutions = new Map<string, ResolvedEntry>()
  const sessions = new Map<string, ReviewSession>()
  const runtimes = new Map<string, Runtime>()
  const startedRequests = new Map<string, { token: string; promise: Promise<unknown> }>()
  const commandResults = new Map<string, unknown>()
  const runs = new Set<Promise<void>>()
  const now = deps.now ?? (() => Date.now())
  const nowMono = deps.nowMono ?? deps.sourceReader.nowMono
  let startQueue = Promise.resolve()

  const currentRecord = (id: string): JournalRecord => deps.journal.read(id)

  const currentSnapshot = (id: string): RolloutSnapshot => validateRolloutSnapshot(currentRecord(id).snapshot)

  const enqueue = <T>(runtime: Runtime, operation: () => Promise<T>): Promise<T> => {
    const next = runtime.queue.then(async () => {
      if (runtime.handoffBarrier) await runtime.handoffBarrier.promise
      return operation()
    }, async () => {
      if (runtime.handoffBarrier) await runtime.handoffBarrier.promise
      return operation()
    })
    runtime.queue = next.then(() => undefined, () => undefined)
    return next
  }

  const persistNow = (
    runtime: Runtime,
    requestId: string,
    payload: unknown,
    state: ManagedRolloutState,
    events: JournalEventInput[],
    facts: JournalEvidenceFact[] = [],
    unresolved?: UnresolvedFenceChange
  ): JournalAck => {
    const current = currentSnapshot(runtime.id)
    const next = snapshotFromState(runtime, state, current)
    const ack = deps.journal.record({
      id: runtime.id,
      expectedRevision: current.revision,
      requestId,
      payload,
      snapshot: next as unknown as JournalSnapshot,
      events,
      facts,
      unresolved
    })
    runtime.snapshot = currentSnapshot(runtime.id)
    return ack
  }

  const persist = async (
    runtime: Runtime,
    requestId: string,
    payload: unknown,
    state: ManagedRolloutState,
    events: JournalEventInput[],
    facts: JournalEvidenceFact[] = [],
    unresolved?: UnresolvedFenceChange
  ): Promise<JournalAck> => enqueue(runtime, async () => persistNow(runtime, requestId, payload, state, events, facts, unresolved))

  const persistAuthorization = async (runtime: Runtime, authorization: ManagedRolloutAuthorization): Promise<void> => {
    await enqueue(runtime, async () => {
      const target = runtime.targets.get(authorization.installId)
      if (!target || target.targetSha !== authorization.targetSha || !sameAuthorization({ ...authorization, reviewedSource: target.reviewedSource }, authorization)) {
        throw new Error('authorization-provenance-mismatch')
      }
      const current = currentSnapshot(runtime.id)
      const attempt = current.attempts.find(item => item.identity.installId === authorization.installId)
      if (!attempt || attempt.correlationId !== authorization.correlationId || attempt.identity.sourceFingerprint !== authorization.sourceFingerprint) {
        throw new Error('authorization-correlation-mismatch')
      }
      const next = clone(current)
      const nextAttempt = next.attempts.find(item => item.identity.installId === authorization.installId)
      if (!nextAttempt) throw new Error('authorization-target-missing')
      nextAttempt.phase = 'updating'
      nextAttempt.launchState = 'authorized'
      nextAttempt.recoveryRequired = false
      nextAttempt.reasons = []
      const recordedAt = isoNow(now)
      const authorizationFence = fence(runtime.id, authorization.installId, authorization.correlationId, recordedAt)
      const ack = deps.journal.record({
        id: runtime.id,
        expectedRevision: current.revision,
        requestId: `authorization:${authorization.installId}:${authorization.correlationId}`,
        payload: {
          kind: 'launch-authorization',
          installId: authorization.installId,
          connectionId: authorization.connectionId,
          sourceFingerprint: authorization.sourceFingerprint,
          targetSha: authorization.targetSha,
          correlationId: authorization.correlationId,
          queueGeneration: authorization.queueGeneration
        },
        snapshot: next as unknown as JournalSnapshot,
        events: [
          event('intent-recorded', authorization.installId),
          event('launch-authorized', authorization.installId)
        ],
        facts: [fact('authorization-committed', runtime.id, authorization.installId, authorization.correlationId, recordedAt, 'coordinator authorization persisted before service handoff')],
        unresolved: { add: [authorizationFence] }
      })
      runtime.snapshot = currentSnapshot(runtime.id)
      runtime.handoffBarrier = createDeferred()
      void ack
    })
  }

  const makeRuntime = (id: string, plan: RolloutPlan): Runtime => {
    const targets = new Map<string, ManagedRolloutTarget>()
    for (const [wave, installIds] of plan.waves.entries()) {
      for (const installId of installIds) {
        const row = plan.rows.find(item => item.installId === installId)
        if (!row?.reviewedSource || !row.requiredScopeIds || !row.admittedHead) throw new Error('rollout-plan-provenance-incomplete')
        targets.set(installId, {
          installId,
          installationFingerprint: row.installationFingerprint,
          connectionId: row.connectionId,
          sourceFingerprint: row.sourceFingerprint,
          targetSha: plan.target.sha,
          reviewedSource: clone(row.reviewedSource),
          correlationId: randomUUID(),
          wave
        })
      }
    }
    const coordinatorPlan = {
      id,
      revision: 1,
      queueGeneration: 0,
      policy: policy(plan.promotionPolicy),
      targets: [...targets.values()]
    }
    const runtime = {
      id,
      plan,
      targets,
      authorizations: new Map<string, ManagedRolloutAuthorization>(),
      observations: new Map<string, ObservationState>(),
      launches: new Map<string, LaunchPromise>(),
      coordinator: null as unknown as Coordinator,
      snapshot: initialSnapshot(id, plan, isoNow(now), new Map([...targets.values()].map(target => [target.installId, target.correlationId]))),
      queue: Promise.resolve(),
      handoffBarrier: null,
      runPromise: null
    } as Runtime

    const coordinatorDependencies: ManagedRolloutCoordinatorDependencies = {
      journal: {
        persistAuthorization: authorization => persistAuthorization(runtime, authorization),
        persistEvent: input => persist(
          runtime,
          `event:${input.kind}:${randomUUID()}`,
          { kind: input.kind, installId: input.installId ?? null, correlationId: input.correlationId ?? null },
          runtime.coordinator.snapshot,
          [event(input.kind, input.installId ?? null, input.reason ?? null)]
        ).then(() => undefined),
      },
      service: {
        issueCapability: authorization => {
          runtime.authorizations.set(authorization.installId, authorization)
          try {
            const capability = deps.managedSshUpdateService.issueLaunchCapability(
              authorization.connectionId,
              authorization.correlationId,
              { targetSha: authorization.targetSha, source: authorization.reviewedSource } satisfies ManagedSshUpdateIntent
            )
            if (!capability || typeof capability !== 'object') throw new Error('invalid-launch-capability')
            return capability as ManagedSshLaunchCapability
          } catch (error) {
            runtime.handoffBarrier?.resolve()
            runtime.handoffBarrier = null
            throw error
          }
        },
        launch: (authorization, capability) => {
          let request: Promise<ManagedConnectionUpdateResult>
          try {
            request = deps.managedSshUpdateService.request(authorization.connectionId, {
              mode: 'coordinator',
              correlationId: authorization.correlationId,
              intent: { targetSha: authorization.targetSha, source: authorization.reviewedSource },
              launchCapability: capability as ManagedSshLaunchCapability
            })
          } catch (error) {
            runtime.handoffBarrier?.resolve()
            runtime.handoffBarrier = null
            throw error
          }
          runtime.handoffBarrier?.resolve()
          runtime.handoffBarrier = null
          const tracked = Promise.resolve(request).then(result => {
            if (!result.ok || !result.updateOk || !result.restoreOk) {
              throw new Error(`managed-update-${result.outcome}`)
            }
            return result
          })
          runtime.launches.set(authorization.installId, tracked)
          return tracked.then(() => undefined)
        }
      },
      evidence: deps.evidence,
      processGeneration: deps.processGeneration ?? 1,
      nowMono,
      recovery: deps.observe.reprobe && deps.observe.recover ? {
        reprobe: authorization => deps.observe.reprobe!(authorization),
        recover: authorization => deps.observe.recover!(authorization)
      } : undefined
    }
    runtime.coordinator = createManagedRolloutCoordinator(createManagedRolloutState(coordinatorPlan), coordinatorDependencies)
    return runtime
  }

  const validateAdmission = async (plan: RolloutPlan): Promise<{
    verifiedSources: ReadonlyMap<string, ReviewedSourceBinding>
    verifiedAssurance: ReadonlyMap<string, VerifiedAssuranceEvidence>
    verifiedInventory: ReadonlyMap<string, VerifiedInventoryRow>
  }> => {
    const verifiedSources = new Map<string, ReviewedSourceBinding>()
    const verifiedAssurance = new Map<string, VerifiedAssuranceEvidence>()
    for (const row of plan.rows) {
      if (!row.reviewedSource || !plan.inventoryRevision) throw new Error('rollout-source-missing')
      const source = await verifyReviewedGitSource(row.reviewedSource, {
        target: plan.target,
        trustedOriginUrl: row.reviewedSource.originUrl,
        repositoryRoot: row.reviewedSource.repositoryRoot,
        branch: plan.target.branch,
        inventoryRevision: plan.inventoryRevision
      }, deps.sourceReader)
      const assurance = await verifyApplicableAssurance({
        profile: source.assuranceProfile,
        repositoryId: plan.target.repositoryId,
        targetSha: plan.target.sha,
        sourceFingerprint: row.sourceFingerprint,
        generation: source.assuranceGeneration,
        now: now()
      }, deps.assuranceReader)
      verifiedSources.set(row.installId, source)
      verifiedAssurance.set(row.installId, assurance)
    }
    const verifiedInventory = await verifyTrustedInventory(plan, nowMono(), deps.inventoryReader)
    return { verifiedSources, verifiedAssurance, verifiedInventory }
  }

  const runRollout = async (runtime: Runtime): Promise<void> => {
    for (let wave = runtime.coordinator.snapshot.currentWave; wave < runtime.plan.waves.length; wave += 1) {
      const beforeWave = runtime.coordinator.snapshot
      if (beforeWave.phase !== 'running' || beforeWave.currentWave !== wave) return
      for (const installId of runtime.plan.waves[wave]) {
        if (runtime.coordinator.snapshot.phase !== 'running') return
        const authorizationTransition = await runtime.coordinator.authorize(installId)
        if (!authorizationTransition.ok) {
          await persist(runtime, `runner-refusal:${installId}:${randomUUID()}`, { kind: 'runner-refusal', installId, reason: authorizationTransition.reason ?? 'unknown' }, authorizationTransition.state, [event('operator-disposition', installId, authorizationTransition.reason ?? 'authorization-refused')])
          return
        }
        const authorization = runtime.authorizations.get(installId)
        const launch = runtime.launches.get(installId)
        if (!authorization || !launch) {
          const failed = await runtime.coordinator.terminal(installId, runtime.targets.get(installId)?.correlationId ?? '', 'unverified')
          await persist(runtime, `runner-missing-launch:${installId}:${randomUUID()}`, { kind: 'runner-missing-launch', installId }, failed.state, [event('attention-required', installId, 'launch-handoff-missing')])
          return
        }
        try {
          const update = await launch
          const observation = await deps.observe.observe({ authorization, update })
          if (observation.authorization && !sameAuthorization(observation.authorization, authorization)) throw new Error('observation-correlation-mismatch')
          if (observation.receipt && observation.receipt.correlationId !== authorization.correlationId) throw new Error('observation-correlation-mismatch')
          const requiredScopeIds = runtime.plan.rows.find(row => row.installId === installId)?.requiredScopeIds ?? []
          if (SUCCESSFUL_OUTCOMES.has(observation.outcome) && (
            !observation.receipt || observation.receipt.correlationId !== authorization.correlationId ||
            !successfulHealth(observation.health, authorization, requiredScopeIds)
          )) {
            throw new Error('successful-observation-not-proven')
          }
          await deps.evidence.recordObservation?.({ authorization, receipt: observation.receipt, health: observation.health })
          runtime.observations.set(installId, { receipt: observation.receipt, health: observation.health })
          const settled = await runtime.coordinator.terminal(installId, authorization.correlationId, observation.outcome)
          const facts = observation.receipt
            ? [
                fact('terminal-receipt', runtime.id, installId, authorization.correlationId, isoNow(now), 'managed SSH service returned a correlated receipt'),
                ...(observation.outcome === 'unverified' ? [] : [fact('settlement-validated', runtime.id, installId, authorization.correlationId, isoNow(now), 'trusted observation validated update and restoration')])
              ]
            : []
          const unresolved: UnresolvedFenceChange | undefined = observation.outcome === 'unverified'
            ? undefined
            : { remove: [fence(runtime.id, installId, authorization.correlationId, isoNow(now)).key] }
          await persist(runtime, `terminal:${installId}:${authorization.correlationId}`, { kind: 'terminal', installId, outcome: observation.outcome, correlationId: authorization.correlationId }, settled.state, [event('launch-observed', installId), event(observation.outcome === 'failed' || observation.outcome === 'refused' ? 'attempt-failed' : 'completed', installId, observation.outcome)], facts, unresolved)
          if (!settled.ok || runtime.coordinator.snapshot.phase !== 'running') return
        } catch (error) {
          const unresolved = await runtime.coordinator.terminal(installId, authorization.correlationId, 'unverified')
          await persist(runtime, `terminal-unverified:${installId}:${authorization.correlationId}`, { kind: 'terminal-unverified', installId, correlationId: authorization.correlationId }, unresolved.state, [event('attention-required', installId, boundedText(error instanceof Error ? error.message : error, 'unverified-launch'))])
          return
        }
      }
      if (runtime.coordinator.snapshot.phase !== 'running') return
    }
  }

  const scheduleRun = (runtime: Runtime): void => {
    if (runtime.runPromise) return
    const run = runRollout(runtime).catch(async () => {
      const state = runtime.coordinator.snapshot
      try {
        await persist(runtime, `runner-error:${randomUUID()}`, { kind: 'runner-error' }, state, [event('attention-required', null, 'runner-failed')])
      } catch {
        // A failed journal write remains a fail-closed unavailable state.
      }
    })
    runtime.runPromise = run
    runs.add(run)
    void run.finally(() => {
      runs.delete(run)
      runtime.runPromise = null
    })
  }

  const startInternal = async (request: { token: string; requestId: string }): Promise<unknown> => {
    const session = sessions.get(request.token)
    if (!session) throw new Error('preflight-token-invalid')
    if (session.requestId !== request.requestId) throw new Error('preflight-request-mismatch')
    if (now() >= session.expiresAt) {
      sessions.delete(request.token)
      reviewTokens.revoke(request.token)
      throw new Error('preflight-token-expired')
    }
    const revalidated = reviewTokens.revalidate(request.token, session.plan, now())
    if (!revalidated.ok) {
      const failed = revalidated as { ok: false; code: string }
      throw new Error(`preflight-${failed.code}`)
    }
    const active = deps.journal.history({ limit: MAX_PROVIDER_PAGE_SIZE }).items.find(item => activePhase(item.phase) && !item.archived)
    if (active) throw new Error('rollout-already-active')
    const runtime = makeRuntime(session.rolloutId, session.plan)
    const startedState = await runtime.coordinator.start(request.requestId)
    const snapshot = runtime.snapshot
    const ack = deps.journal.create(snapshot as unknown as JournalSnapshot, {
      request: { requestId: request.requestId, payload: { kind: 'start', planDigest: session.planDigest } },
      events: [event('created')]
    })
    runtime.snapshot = currentSnapshot(runtime.id)
    runtimes.set(runtime.id, runtime)
    sessions.delete(request.token)
    reviewTokens.revoke(request.token)
    const result = { ok: true, id: runtime.id, revision: ack.revision, phase: 'running' as const }
    scheduleRun(runtime)
    void startedState
    return result
  }

  const command = async (raw: ManagedRolloutIpcCommand): Promise<unknown> => {
    const parsed = normalizedCommand(raw)
    const record = currentRecord(parsed.id)
    const payload = {
      action: parsed.action,
      installId: parsed.installId,
      reason: parsed.reason,
      promotionPolicy: parsed.promotionPolicy,
      expectedRevision: parsed.expectedRevision
    }
    const existing = record.requests[parsed.requestId]
    if (existing) {
      if (existing.payloadDigest !== payloadDigest(payload)) return staleAck(parsed.id, record.snapshot.revision, 'request-payload-mismatch', 'requestId was reused with a different command.')
      return commandResults.get(parsed.requestId) ?? {
        ok: true,
        id: parsed.id,
        revision: existing.ack.revision,
        code: 'duplicate',
        message: null,
        changes: []
      }
    }
    const current = validateRolloutSnapshot(record.snapshot)
    if (parsed.expectedRevision !== current.revision) return staleAck(parsed.id, current.revision, 'stale-revision', 'managed rollout revision is stale.')
    if (parsed.action === 'archive') {
      if (!TERMINAL_PHASES.has(current.phase)) return staleAck(parsed.id, current.revision, 'archive-not-admissible', 'only terminal rollouts can be archived.')
      const ack = deps.journal.archive({
        id: parsed.id,
        expectedRevision: current.revision,
        requestId: parsed.requestId,
        payload,
        actor: 'local-operator',
        reason: parsed.reason || 'archived'
      })
      const result = { ok: true, id: parsed.id, revision: ack.revision, code: null, message: null, changes: [] as PlanChange[] }
      commandResults.set(parsed.requestId, result)
      return result
    }
    const runtime = runtimes.get(parsed.id)
    if (!runtime) return staleAck(parsed.id, current.revision, 'runtime-not-hydrated', 'this rollout has journal evidence but no safe in-process coordinator.')

    return enqueue(runtime, async () => {
      const liveRecord = currentRecord(parsed.id)
      const liveExisting = liveRecord.requests[parsed.requestId]
      if (liveExisting) {
        if (liveExisting.payloadDigest !== payloadDigest(payload)) return staleAck(parsed.id, liveRecord.snapshot.revision, 'request-payload-mismatch', 'requestId was reused with a different command.')
        return commandResults.get(parsed.requestId) ?? {
          ok: true,
          id: parsed.id,
          revision: liveExisting.ack.revision,
          code: 'duplicate',
          message: null,
          changes: []
        }
      }
      const liveCurrent = validateRolloutSnapshot(liveRecord.snapshot)
      if (parsed.expectedRevision !== liveCurrent.revision) return staleAck(parsed.id, liveCurrent.revision, 'stale-revision', 'managed rollout revision is stale.')

      let transition: { ok: boolean; state: ManagedRolloutState; reason?: string }
      switch (parsed.action) {
        case 'pause': transition = await runtime.coordinator.command({ kind: 'pause' }); break
        case 'resume': transition = await runtime.coordinator.command({ kind: 'resume' }); break
        case 'stop': transition = await runtime.coordinator.command({ kind: 'stop' }); break
        case 'exclude': transition = await runtime.coordinator.command({ kind: 'exclude', installId: parsed.installId! }); break
        case 'promote': transition = await runtime.coordinator.promote(false); break
        case 'reprobe': transition = await runtime.coordinator.reprobe(parsed.installId!); break
        case 'recover': transition = await runtime.coordinator.recover(parsed.installId!); break
        case 'set-policy': return staleAck(parsed.id, liveCurrent.revision, 'policy-change-unsupported', 'policy changes are not safe without a coordinator reconstruction.')
        default: return staleAck(parsed.id, liveCurrent.revision, 'unsupported-command', 'managed rollout command is not supported.')
      }
      const reason = transition.reason ?? null
      const ack = persistNow(runtime, parsed.requestId, payload, transition.state, [event(commandEventKind(parsed.action, transition.ok), parsed.installId, reason, 'local-operator')])
      const result = {
        ok: transition.ok,
        id: parsed.id,
        revision: ack.revision,
        code: transition.ok ? null : reason,
        message: transition.ok ? null : reason,
        changes: [] as PlanChange[]
      }
      commandResults.set(parsed.requestId, result)
      if (transition.ok && (parsed.action === 'resume' || parsed.action === 'promote')) scheduleRun(runtime)
      return result
    })
  }

  const provider: ManagedRolloutProvider = {
    capabilities: async () => capabilities(deps.ready ? deps.ready() : true),
    inventory: async () => {
      const snapshot = await deps.inventoryReader.capture()
      if (!snapshot) throw new Error('inventory-unavailable')
      return inventoryOutput(snapshot)
    },
    resolveTarget: async request => {
      if (!Array.isArray(request.connectionIds) || request.connectionIds.length === 0 || request.connectionIds.length > 500) throw new Error('target-connection-list-invalid')
      if (typeof request.inventoryRevision !== 'string' || !request.inventoryRevision || request.inventoryRevision.length > 256) throw new Error('inventory-revision-invalid')
      const resolved = await deps.resolveTarget(request)
      const plan = validateRolloutPlan(resolved.plan)
      const resolution = validateTargetResolution(resolved.resolution)
      if (plan.inventoryRevision !== request.inventoryRevision || plan.retryOf !== request.retryOf || !exactConnectionSet(plan, request.connectionIds) || JSON.stringify(plan.target) !== JSON.stringify(resolution.target)) {
        throw new Error('target-resolution-provenance-mismatch')
      }
      await validateAdmission(plan)
      const prior = resolutions.get(resolution.id)
      if (prior && canonicalPlanDigest(prior.plan) !== canonicalPlanDigest(plan)) throw new Error('target-resolution-reuse-mismatch')
      resolutions.set(resolution.id, { plan: clone(plan), resolution: clone(resolution) })
      return targetResolutionOutput({ plan, resolution })
    },
    preflight: async (rawDraft: unknown) => {
      if (!rawDraft || typeof rawDraft !== 'object' || Array.isArray(rawDraft)) throw new Error('preflight-draft-invalid')
      const draft = rawDraft as Record<string, unknown>
      const keys = ['inventoryRevision', 'targetResolutionId', 'waves', 'concurrency', 'promotionPolicy', 'retryOf']
      if (Object.keys(draft).sort().join('|') !== keys.slice().sort().join('|')) throw new Error('preflight-draft-invalid')
      if (typeof draft.targetResolutionId !== 'string' || typeof draft.inventoryRevision !== 'string' || !Array.isArray(draft.waves) || !Number.isSafeInteger(draft.concurrency) || (draft.promotionPolicy !== 'manual' && draft.promotionPolicy !== 'auto-if-healthy') || (draft.retryOf !== null && typeof draft.retryOf !== 'string')) throw new Error('preflight-draft-invalid')
      const entry = resolutions.get(draft.targetResolutionId)
      if (!entry) throw new Error('target-resolution-unavailable')
      if (now() >= entry.resolution.expiresAt) throw new Error('target-resolution-expired')
      const members = (draft.waves as unknown[]).map(wave => {
        if (!Array.isArray(wave) || wave.some(installId => typeof installId !== 'string')) throw new Error('preflight-draft-invalid')
        return [...wave] as string[]
      })
      const memberSet = new Set(members.flat())
      if (memberSet.size !== members.flat().length || members.length === 0 || members.some(wave => wave.length === 0)) throw new Error('preflight-draft-invalid')
      const exclusions = entry.plan.rows.map(row => row.installId).filter(installId => !memberSet.has(installId))
      const plan = validateRolloutPlan({
        target: entry.plan.target,
        inventoryRevision: draft.inventoryRevision,
        waves: members,
        concurrency: draft.concurrency,
        promotionPolicy: draft.promotionPolicy,
        rows: entry.plan.rows,
        retryOf: draft.retryOf,
        exclusions
      })
      if (plan.inventoryRevision !== entry.plan.inventoryRevision || plan.retryOf !== entry.plan.retryOf) throw new Error('preflight-plan-provenance-mismatch')
      const admission = await validateAdmission(plan)
      const review = createPreflightReview({
        plan,
        resolution: entry.resolution,
        reviewTokens,
        now: now(),
        nowMono: nowMono(),
        verifiedSources: admission.verifiedSources,
        verifiedAssurance: admission.verifiedAssurance,
        verifiedInventory: admission.verifiedInventory
      })
      if (!review.token || !review.planDigest || !review.expiresAt) {
        return { ok: false, token: null, requestId: null, rolloutId: null, expiresAt: null, planDigest: null, canonicalPlan: clone(review.canonicalPlan), changes: review.changes, blockers: review.blockers }
      }
      const requestId = randomUUID()
      const rolloutId = randomUUID()
      sessions.set(review.token, {
        token: review.token,
        requestId,
        rolloutId,
        plan: clone(review.canonicalPlan),
        expiresAt: review.expiresAt,
        planDigest: review.planDigest
      })
      return { ok: true, token: review.token, requestId, rolloutId, expiresAt: review.expiresAt, planDigest: review.planDigest, canonicalPlan: clone(review.canonicalPlan), changes: review.changes, blockers: review.blockers }
    },
    start: async request => {
      const existing = startedRequests.get(request.requestId)
      if (existing) {
        if (existing.token !== request.token) throw new Error('requestId-token-mismatch')
        return existing.promise
      }
      const promise = startQueue.then(() => startInternal(request), () => startInternal(request))
      startQueue = promise.then(() => undefined, () => undefined)
      startedRequests.set(request.requestId, { token: request.token, promise })
      return promise
    },
    activeRevision: async () => {
      const page = deps.journal.history({ limit: MAX_PROVIDER_PAGE_SIZE })
      const active = page.items.filter(item => activePhase(item.phase) && !item.archived)
      return active.length ? Math.max(...active.map(item => item.revision)) : null
    },
    read: async sinceRevision => {
      if (sinceRevision !== null && (!Number.isSafeInteger(sinceRevision) || sinceRevision < 0)) throw new Error('managed-rollout-revision-invalid')
      const page = deps.journal.history({ limit: MAX_PROVIDER_PAGE_SIZE })
      const active = page.items
        .filter(item => activePhase(item.phase) && !item.archived)
        .sort((left, right) => right.revision - left.revision)
      if (!active.length) return { revision: 0, snapshot: null }
      const current = active[0]
      if (sinceRevision === current.revision) return { revision: current.revision, snapshot: null }
      return { revision: current.revision, snapshot: validateRolloutSnapshot(deps.journal.read(current.id).snapshot) }
    },
    get: async id => {
      try {
        return validateRolloutSnapshot(deps.journal.read(id).snapshot)
      } catch (error) {
        if (error instanceof Error && /not found/i.test(error.message)) return null
        throw error
      }
    },
    command,
    history: async page => {
      if (!Number.isSafeInteger(page.limit) || page.limit < 1 || page.limit > MAX_PROVIDER_PAGE_SIZE) throw new Error('history-page-limit-invalid')
      return deps.journal.history({ cursor: page.cursor ?? null, limit: page.limit })
    },
    events: async page => {
      if (!Number.isSafeInteger(page.limit) || page.limit < 1 || page.limit > MAX_PROVIDER_PAGE_SIZE) throw new Error('events-page-limit-invalid')
      return deps.journal.events(page.id, { cursor: page.cursor ?? null, limit: page.limit })
    },
    waitForIdle: async () => {
      while (runs.size) await Promise.all([...runs])
    }
  }
  return provider
}
