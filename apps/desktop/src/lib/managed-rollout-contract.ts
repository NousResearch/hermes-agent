export type PromotionPolicy = 'manual' | 'auto-if-healthy'

export type RolloutPhase =
  | 'running'
  | 'pausing'
  | 'paused'
  | 'awaiting-promotion'
  | 'attention-required'
  | 'reconciling'
  | 'stopping'
  | 'stopped'
  | 'completed'
  | 'completed-with-exclusions'

export type TargetPhase =
  | 'queued'
  | 'preflight'
  | 'draining'
  | 'updating'
  | 'awaiting-receipt'
  | 'restoring'
  | 'verifying'
  | 'updated'
  | 'already-current'
  | 'failed'
  | 'refused'
  | 'unverified'
  | 'recovery-required'
  | 'skipped'

export type LaunchState = 'none' | 'intent-recorded' | 'authorized' | 'observed'

export type SkipReason = 'stop-recorded' | 'cancelled-before-launch' | 'operator-excluded'

export type EventKind =
  | 'created'
  | 'intent-recorded'
  | 'launch-authorized'
  | 'launch-observed'
  | 'wave-started'
  | 'wave-settled'
  | 'promoted'
  | 'auto-promoted'
  | 'policy-changed'
  | 'pause-requested'
  | 'paused'
  | 'resumed'
  | 'stop-requested'
  | 'stopped'
  | 'attention-required'
  | 'reprobed'
  | 'attempt-excluded'
  | 'recovered'
  | 'operator-disposition'
  | 'archived'
  | 'reconciled'
  | 'attempt-failed'
  | 'completed'

export type RolloutAction =
  'promote' | 'pause' | 'resume' | 'stop' | 'reprobe' | 'recover' | 'exclude' | 'archive' | 'set-policy'

export const PROMOTION_POLICIES = ['manual', 'auto-if-healthy'] as const
export const ROLLOUT_PHASES = [
  'running',
  'pausing',
  'paused',
  'awaiting-promotion',
  'attention-required',
  'reconciling',
  'stopping',
  'stopped',
  'completed',
  'completed-with-exclusions'
] as const
export const TARGET_PHASES = [
  'queued',
  'preflight',
  'draining',
  'updating',
  'awaiting-receipt',
  'restoring',
  'verifying',
  'updated',
  'already-current',
  'failed',
  'refused',
  'unverified',
  'recovery-required',
  'skipped'
] as const
export const LAUNCH_STATES = ['none', 'intent-recorded', 'authorized', 'observed'] as const
export const SKIP_REASONS = ['stop-recorded', 'cancelled-before-launch', 'operator-excluded'] as const
export const EVENT_KINDS = [
  'created',
  'intent-recorded',
  'launch-authorized',
  'launch-observed',
  'wave-started',
  'wave-settled',
  'promoted',
  'auto-promoted',
  'policy-changed',
  'pause-requested',
  'paused',
  'resumed',
  'stop-requested',
  'stopped',
  'attention-required',
  'reprobed',
  'attempt-excluded',
  'recovered',
  'operator-disposition',
  'archived',
  'reconciled',
  'attempt-failed',
  'completed'
] as const
export const ROLLOUT_ACTIONS = [
  'promote',
  'pause',
  'resume',
  'stop',
  'reprobe',
  'recover',
  'exclude',
  'archive',
  'set-policy'
] as const

export const MAX_ROLLOUT_INSTALLATIONS = 500
export const MAX_ROLLOUT_CONCURRENCY = 4

export interface RolloutCapabilities {
  protocol: 1
  available: boolean
  reason: string | null
  maxConcurrency: number
  maxInstallations: number
}

export interface TargetIdentity {
  connectionId: string
  installId: string
  aliasConnectionIds: string[]
  label: string
  displayAddress: string
  installationFingerprint: string
  sourceFingerprint: string
  admittedSha: string
}

export interface RolloutTarget {
  repositoryId: string
  branch: string
  sha: string
  protocol: 1
}

/** Wire identity consumed by the pinned Python updater. Review is separate. */
export interface ReviewedSourceBinding {
  repositoryRoot: string
  originUrl: string
  resolvedRef: string
  targetSha: string
  assuranceProfile: string
  assuranceEvidenceSha256: string
  assuranceGeneration: number
}

export interface ScopeEvidence {
  scopeId: string
  profile: string
  restored: boolean
  ready: boolean
  codeSha: string | null
  processIdentityVerified: boolean
}

export type ScopeCapture = 'complete' | 'missing'

export interface HealthEvidence {
  observationId: string
  observedAt: string
  installId: string | null
  checkoutSha: string | null
  installReady: boolean
  markerClear: boolean
  receiptCorrelated: boolean
  receiptSucceeded: boolean
  dependencyReady: boolean
  recoveryClear: boolean
  scopeCapture: ScopeCapture
  scopes: ScopeEvidence[]
  reasons: string[]
}

export interface ReceiptSummary {
  correlationId: string
  installId: string | null
  requestedSha: string | null
  preSha: string | null
  postSha: string | null
  outcome: string
  startedAt: string | null
  finishedAt: string | null
  stopReason: string | null
}

export interface TargetAttempt {
  identity: TargetIdentity
  correlationId: string
  wave: number
  phase: TargetPhase
  launchState: LaunchState
  requiredScopeIds: string[] | null
  skipReason: SkipReason | null
  reprobes: number
  receipt: ReceiptSummary | null
  health: HealthEvidence | null
  recoveryRequired: boolean
  reasons: string[]
}

export interface RolloutEvent {
  sequence: number
  at: string
  kind: EventKind
  actor: 'local-operator' | 'system'
  installId: string | null
  reason: string | null
  evidenceDigest: string | null
}

export interface RolloutSnapshot {
  schemaVersion: 1
  id: string
  revision: number
  createdAt: string
  updatedAt: string
  finishedAt: string | null
  retryOf: string | null
  archivedAt: string | null
  target: RolloutTarget
  phase: RolloutPhase
  activeWave: number
  concurrency: number
  promotionPolicy: PromotionPolicy
  canaryApproved: boolean
  continuationRequired: boolean
  attempts: TargetAttempt[]
  eventCount: number
}

export interface RolloutDraft {
  inventoryRevision: string
  targetResolutionId: string
  waves: string[][]
  concurrency: number
  promotionPolicy: PromotionPolicy
  retryOf: string | null
}

export type PlanChangeField = 'membership' | 'identity' | 'source' | 'head' | 'scopes' | 'eligibility'

export interface PlanChange {
  installId: string
  field: PlanChangeField
  before: string | null
  after: string | null
}

export interface RolloutPlanRow {
  installId: string
  connectionId: string
  installationFingerprint: string
  sourceFingerprint: string
  admittedHead: string | null
  requiredScopeIds: string[] | null
  eligible: boolean
  reviewedSource?: ReviewedSourceBinding
}

/** The main-owned canonical plan used by preflight/revalidation. */
export interface RolloutPlan {
  target: RolloutTarget
  inventoryRevision?: string
  waves: string[][]
  concurrency: number
  promotionPolicy: PromotionPolicy
  rows: RolloutPlanRow[]
  retryOf: string | null
  exclusions: string[]
}

export interface CommandAck {
  ok: boolean
  id: string | null
  revision: number | null
  code: string | null
  message: string | null
  changes: PlanChange[]
}

export interface RevisionSummary {
  id: string
  revision: number
  phase: RolloutPhase
  updatedAt: string
}

export interface RolloutCommand {
  id: string
  expectedRevision: number
  requestId: string
  action: RolloutAction
  installId: string | null
  reason: string | null
  promotionPolicy: PromotionPolicy | null
}

export class RolloutContractError extends Error {
  readonly code = 'invalid-rollout-contract'
  readonly path: string

  constructor(path: string, message: string) {
    super(`${path}: ${message}`)
    this.name = 'RolloutContractError'
    this.path = path
  }
}

const SHA_RE = /^[0-9a-f]{40}$/
const INSTALL_ID_RE = /^[0-9a-f]{32}$/
const FINGERPRINT_RE = /^[0-9a-f]{64}$/

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function fail(path: string, message: string): never {
  throw new RolloutContractError(path, message)
}

function exactKeys(value: Record<string, unknown>, keys: readonly string[], path: string): void {
  const allowed = new Set(keys)
  const actual = Object.keys(value)
  const missing = keys.filter(key => !Object.prototype.hasOwnProperty.call(value, key))
  const extra = actual.filter(key => !allowed.has(key))

  if (missing.length) fail(path, `missing authority field ${missing.join(', ')}`)
  if (extra.length) fail(path, `unknown field ${extra.join(', ')}`)
}

function stringValue(value: unknown, path: string, allowEmpty = false): string {
  if (typeof value !== 'string' || (!allowEmpty && value.length === 0) || /[\x00\r\n]/.test(value)) {
    fail(path, 'expected a bounded string')
  }

  if (value.length > 4096) fail(path, 'string is too long')

  return value
}

function nullableString(value: unknown, path: string): string | null {
  return value === null ? null : stringValue(value, path)
}

function booleanValue(value: unknown, path: string): boolean {
  if (typeof value !== 'boolean') fail(path, 'expected boolean')

  return value
}

function integerValue(value: unknown, path: string, minimum = 0): number {
  if (!Number.isSafeInteger(value) || (value as number) < minimum) fail(path, `expected integer >= ${minimum}`)

  return value as number
}

function enumValue<T extends string>(value: unknown, values: readonly T[], path: string): T {
  if (typeof value !== 'string' || !values.includes(value as T)) fail(path, 'unsupported enum value')

  return value as T
}

function shaValue(value: unknown, path: string, nullable = false): string | null {
  if (nullable && value === null) return null
  if (typeof value !== 'string' || !SHA_RE.test(value)) fail(path, 'expected lowercase 40-character SHA')

  return value
}

function installIdValue(value: unknown, path: string): string {
  if (typeof value !== 'string' || !INSTALL_ID_RE.test(value)) fail(path, 'expected lowercase 32-character install id')

  return value
}

function fingerprintValue(value: unknown, path: string): string {
  if (typeof value !== 'string' || !FINGERPRINT_RE.test(value)) fail(path, 'expected lowercase SHA-256 fingerprint')

  return value
}

function stringArray(value: unknown, path: string, unique = false): string[] {
  if (!Array.isArray(value)) fail(path, 'expected array')

  const result = value.map((entry, index) => stringValue(entry, `${path}[${index}]`))

  if (unique && new Set(result).size !== result.length) fail(path, 'duplicate values are not allowed')

  return result
}

function nullableStringArray(value: unknown, path: string): string[] | null {
  return value === null ? null : stringArray(value, path, true)
}

function isoString(value: unknown, path: string): string {
  const result = stringValue(value, path)

  if (Number.isNaN(Date.parse(result))) fail(path, 'expected ISO timestamp')

  return result
}

export function isPromotionPolicy(value: unknown): value is PromotionPolicy {
  return typeof value === 'string' && PROMOTION_POLICIES.includes(value as PromotionPolicy)
}

export function isRolloutPhase(value: unknown): value is RolloutPhase {
  return typeof value === 'string' && ROLLOUT_PHASES.includes(value as RolloutPhase)
}

export function isTargetPhase(value: unknown): value is TargetPhase {
  return typeof value === 'string' && TARGET_PHASES.includes(value as TargetPhase)
}

export function validateRolloutCapabilities(value: unknown): RolloutCapabilities {
  if (!isRecord(value)) fail('capabilities', 'expected object')
  exactKeys(value, ['protocol', 'available', 'reason', 'maxConcurrency', 'maxInstallations'], 'capabilities')
  if (value.protocol !== 1) fail('capabilities.protocol', 'unsupported protocol')

  const maxConcurrency = integerValue(value.maxConcurrency, 'capabilities.maxConcurrency', 0)
  const maxInstallations = integerValue(value.maxInstallations, 'capabilities.maxInstallations', 0)

  if (maxConcurrency > MAX_ROLLOUT_CONCURRENCY || maxInstallations > MAX_ROLLOUT_INSTALLATIONS) {
    fail('capabilities', 'capability exceeds release bounds')
  }
  if (!value.available && maxConcurrency !== 0)
    fail('capabilities.maxConcurrency', 'unavailable capability must advertise zero concurrency')
  if (value.available && (maxConcurrency < 1 || maxInstallations < 1))
    fail('capabilities', 'available capability must advertise usable capacity')

  return {
    protocol: 1,
    available: booleanValue(value.available, 'capabilities.available'),
    reason: nullableString(value.reason, 'capabilities.reason'),
    maxConcurrency,
    maxInstallations
  }
}

export function validateTargetIdentity(value: unknown, path = 'identity'): TargetIdentity {
  if (!isRecord(value)) fail(path, 'expected object')
  exactKeys(
    value,
    [
      'connectionId',
      'installId',
      'aliasConnectionIds',
      'label',
      'displayAddress',
      'installationFingerprint',
      'sourceFingerprint',
      'admittedSha'
    ],
    path
  )

  return {
    connectionId: stringValue(value.connectionId, `${path}.connectionId`),
    installId: installIdValue(value.installId, `${path}.installId`),
    aliasConnectionIds: stringArray(value.aliasConnectionIds, `${path}.aliasConnectionIds`, true),
    label: stringValue(value.label, `${path}.label`),
    displayAddress: stringValue(value.displayAddress, `${path}.displayAddress`),
    installationFingerprint: fingerprintValue(value.installationFingerprint, `${path}.installationFingerprint`),
    sourceFingerprint: fingerprintValue(value.sourceFingerprint, `${path}.sourceFingerprint`),
    admittedSha: shaValue(value.admittedSha, `${path}.admittedSha`) as string
  }
}

export function validateRolloutTarget(value: unknown, path = 'target'): RolloutTarget {
  if (!isRecord(value)) fail(path, 'expected object')
  exactKeys(value, ['repositoryId', 'branch', 'sha', 'protocol'], path)
  if (value.protocol !== 1) fail(`${path}.protocol`, 'unsupported protocol')

  return {
    repositoryId: stringValue(value.repositoryId, `${path}.repositoryId`),
    branch: stringValue(value.branch, `${path}.branch`),
    sha: shaValue(value.sha, `${path}.sha`) as string,
    protocol: 1
  }
}

export function validateReviewedSourceBinding(value: unknown, path = 'reviewedSource'): ReviewedSourceBinding {
  if (!isRecord(value)) fail(path, 'expected object')
  exactKeys(value, [
    'repositoryRoot', 'originUrl', 'resolvedRef', 'targetSha',
    'assuranceProfile', 'assuranceEvidenceSha256', 'assuranceGeneration'
  ], path)
  const root = stringValue(value.repositoryRoot, `${path}.repositoryRoot`)
  const origin = stringValue(value.originUrl, `${path}.originUrl`)
  const ref = stringValue(value.resolvedRef, `${path}.resolvedRef`)
  const profile = stringValue(value.assuranceProfile, `${path}.assuranceProfile`)
  if (!(/^(?:[A-Za-z]:[\\/]|\/)/.test(root)) || /[\x00-\x1f\x7f]/.test(root))
    fail(`${path}.repositoryRoot`, 'expected absolute path')
  if (/\s|[\x00-\x1f\x7f]|[?#]/.test(origin)) fail(`${path}.originUrl`, 'credential-free remote required')
  if (!ref.startsWith('refs/remotes/origin/') || ref.endsWith('/') || ref.includes('..') || ref.includes('//'))
    fail(`${path}.resolvedRef`, 'expected resolved origin ref')
  if (/\s|[\x00-\x1f\x7f]/.test(ref) || /\s|[\x00-\x1f\x7f]/.test(profile))
    fail(path, 'source contains control or whitespace')
  return {
    repositoryRoot: root,
    originUrl: origin,
    resolvedRef: ref,
    targetSha: shaValue(value.targetSha, `${path}.targetSha`) as string,
    assuranceProfile: profile,
    assuranceEvidenceSha256: fingerprintValue(value.assuranceEvidenceSha256, `${path}.assuranceEvidenceSha256`),
    assuranceGeneration: integerValue(value.assuranceGeneration, `${path}.assuranceGeneration`)
  }
}

export function validateScopeEvidence(value: unknown, path = 'scope'): ScopeEvidence {
  if (!isRecord(value)) fail(path, 'expected object')
  exactKeys(value, ['scopeId', 'profile', 'restored', 'ready', 'codeSha', 'processIdentityVerified'], path)

  return {
    scopeId: stringValue(value.scopeId, `${path}.scopeId`),
    profile: stringValue(value.profile, `${path}.profile`),
    restored: booleanValue(value.restored, `${path}.restored`),
    ready: booleanValue(value.ready, `${path}.ready`),
    codeSha: shaValue(value.codeSha, `${path}.codeSha`, true),
    processIdentityVerified: booleanValue(value.processIdentityVerified, `${path}.processIdentityVerified`)
  }
}

export function validateHealthEvidence(value: unknown, path = 'health'): HealthEvidence {
  if (!isRecord(value)) fail(path, 'expected object')
  exactKeys(
    value,
    [
      'observationId',
      'observedAt',
      'installId',
      'checkoutSha',
      'installReady',
      'markerClear',
      'receiptCorrelated',
      'receiptSucceeded',
      'dependencyReady',
      'recoveryClear',
      'scopeCapture',
      'scopes',
      'reasons'
    ],
    path
  )

  const scopeCapture = enumValue(value.scopeCapture, ['complete', 'missing'], `${path}.scopeCapture`)
  if (!Array.isArray(value.scopes)) fail(`${path}.scopes`, 'missing scope inventory')
  const scopes = value.scopes.map((entry, index) => validateScopeEvidence(entry, `${path}.scopes[${index}]`))
  if (new Set(scopes.map(scope => scope.scopeId)).size !== scopes.length) fail(`${path}.scopes`, 'duplicate scope id')
  if (scopeCapture === 'missing' && scopes.length !== 0)
    fail(`${path}.scopes`, 'missing capture must not contain scopes')

  return {
    observationId: stringValue(value.observationId, `${path}.observationId`),
    observedAt: isoString(value.observedAt, `${path}.observedAt`),
    installId: value.installId === null ? null : installIdValue(value.installId, `${path}.installId`),
    checkoutSha: shaValue(value.checkoutSha, `${path}.checkoutSha`, true),
    installReady: booleanValue(value.installReady, `${path}.installReady`),
    markerClear: booleanValue(value.markerClear, `${path}.markerClear`),
    receiptCorrelated: booleanValue(value.receiptCorrelated, `${path}.receiptCorrelated`),
    receiptSucceeded: booleanValue(value.receiptSucceeded, `${path}.receiptSucceeded`),
    dependencyReady: booleanValue(value.dependencyReady, `${path}.dependencyReady`),
    recoveryClear: booleanValue(value.recoveryClear, `${path}.recoveryClear`),
    scopeCapture,
    scopes,
    reasons: stringArray(value.reasons, `${path}.reasons`)
  }
}

export function validateReceiptSummary(value: unknown, path = 'receipt'): ReceiptSummary {
  if (!isRecord(value)) fail(path, 'expected object')
  exactKeys(
    value,
    [
      'correlationId',
      'installId',
      'requestedSha',
      'preSha',
      'postSha',
      'outcome',
      'startedAt',
      'finishedAt',
      'stopReason'
    ],
    path
  )

  return {
    correlationId: stringValue(value.correlationId, `${path}.correlationId`),
    installId: value.installId === null ? null : installIdValue(value.installId, `${path}.installId`),
    requestedSha: shaValue(value.requestedSha, `${path}.requestedSha`, true),
    preSha: shaValue(value.preSha, `${path}.preSha`, true),
    postSha: shaValue(value.postSha, `${path}.postSha`, true),
    outcome: stringValue(value.outcome, `${path}.outcome`),
    startedAt: value.startedAt === null ? null : isoString(value.startedAt, `${path}.startedAt`),
    finishedAt: value.finishedAt === null ? null : isoString(value.finishedAt, `${path}.finishedAt`),
    stopReason: nullableString(value.stopReason, `${path}.stopReason`)
  }
}

export function validateTargetAttempt(value: unknown, path = 'attempt'): TargetAttempt {
  if (!isRecord(value)) fail(path, 'expected object')
  exactKeys(
    value,
    [
      'identity',
      'correlationId',
      'wave',
      'phase',
      'launchState',
      'requiredScopeIds',
      'skipReason',
      'reprobes',
      'receipt',
      'health',
      'recoveryRequired',
      'reasons'
    ],
    path
  )

  return {
    identity: validateTargetIdentity(value.identity, `${path}.identity`),
    correlationId: stringValue(value.correlationId, `${path}.correlationId`),
    wave: integerValue(value.wave, `${path}.wave`),
    phase: enumValue(value.phase, TARGET_PHASES, `${path}.phase`),
    launchState: enumValue(value.launchState, LAUNCH_STATES, `${path}.launchState`),
    requiredScopeIds: nullableStringArray(value.requiredScopeIds, `${path}.requiredScopeIds`),
    skipReason: value.skipReason === null ? null : enumValue(value.skipReason, SKIP_REASONS, `${path}.skipReason`),
    reprobes: integerValue(value.reprobes, `${path}.reprobes`),
    receipt: value.receipt === null ? null : validateReceiptSummary(value.receipt, `${path}.receipt`),
    health: value.health === null ? null : validateHealthEvidence(value.health, `${path}.health`),
    recoveryRequired: booleanValue(value.recoveryRequired, `${path}.recoveryRequired`),
    reasons: stringArray(value.reasons, `${path}.reasons`)
  }
}

export function validateRolloutEvent(value: unknown, path = 'event'): RolloutEvent {
  if (!isRecord(value)) fail(path, 'expected object')
  exactKeys(value, ['sequence', 'at', 'kind', 'actor', 'installId', 'reason', 'evidenceDigest'], path)

  return {
    sequence: integerValue(value.sequence, `${path}.sequence`),
    at: isoString(value.at, `${path}.at`),
    kind: enumValue(value.kind, EVENT_KINDS, `${path}.kind`),
    actor: enumValue(value.actor, ['local-operator', 'system'], `${path}.actor`),
    installId: value.installId === null ? null : installIdValue(value.installId, `${path}.installId`),
    reason: nullableString(value.reason, `${path}.reason`),
    evidenceDigest:
      value.evidenceDigest === null ? null : fingerprintValue(value.evidenceDigest, `${path}.evidenceDigest`)
  }
}

export function validateRolloutSnapshot(value: unknown): RolloutSnapshot {
  if (!isRecord(value)) fail('snapshot', 'expected object')
  exactKeys(
    value,
    [
      'schemaVersion',
      'id',
      'revision',
      'createdAt',
      'updatedAt',
      'finishedAt',
      'retryOf',
      'archivedAt',
      'target',
      'phase',
      'activeWave',
      'concurrency',
      'promotionPolicy',
      'canaryApproved',
      'continuationRequired',
      'attempts',
      'eventCount'
    ],
    'snapshot'
  )
  if (value.schemaVersion !== 1) fail('snapshot.schemaVersion', 'unsupported schema')
  if (!Array.isArray(value.attempts)) fail('snapshot.attempts', 'expected array')

  const attempts = value.attempts.map((entry, index) => validateTargetAttempt(entry, `snapshot.attempts[${index}]`))
  if (new Set(attempts.map(attempt => attempt.identity.installId)).size !== attempts.length) {
    fail('snapshot.attempts', 'duplicate installation')
  }

  return {
    schemaVersion: 1,
    id: stringValue(value.id, 'snapshot.id'),
    revision: integerValue(value.revision, 'snapshot.revision'),
    createdAt: isoString(value.createdAt, 'snapshot.createdAt'),
    updatedAt: isoString(value.updatedAt, 'snapshot.updatedAt'),
    finishedAt: value.finishedAt === null ? null : isoString(value.finishedAt, 'snapshot.finishedAt'),
    retryOf: nullableString(value.retryOf, 'snapshot.retryOf'),
    archivedAt: value.archivedAt === null ? null : isoString(value.archivedAt, 'snapshot.archivedAt'),
    target: validateRolloutTarget(value.target),
    phase: enumValue(value.phase, ROLLOUT_PHASES, 'snapshot.phase'),
    activeWave: integerValue(value.activeWave, 'snapshot.activeWave'),
    concurrency: integerValue(value.concurrency, 'snapshot.concurrency'),
    promotionPolicy: enumValue(value.promotionPolicy, PROMOTION_POLICIES, 'snapshot.promotionPolicy'),
    canaryApproved: booleanValue(value.canaryApproved, 'snapshot.canaryApproved'),
    continuationRequired: booleanValue(value.continuationRequired, 'snapshot.continuationRequired'),
    attempts,
    eventCount: integerValue(value.eventCount, 'snapshot.eventCount')
  }
}

function validatePlanRow(value: unknown, path: string): RolloutPlanRow {
  if (!isRecord(value)) fail(path, 'expected object')
  exactKeys(
    value,
    [
      'installId',
      'connectionId',
      'installationFingerprint',
      'sourceFingerprint',
      'admittedHead',
      'requiredScopeIds',
      'eligible',
      ...(Object.prototype.hasOwnProperty.call(value, 'reviewedSource') ? ['reviewedSource'] : [])
    ],
    path
  )

  return {
    installId: installIdValue(value.installId, `${path}.installId`),
    connectionId: stringValue(value.connectionId, `${path}.connectionId`),
    installationFingerprint: fingerprintValue(value.installationFingerprint, `${path}.installationFingerprint`),
    sourceFingerprint: fingerprintValue(value.sourceFingerprint, `${path}.sourceFingerprint`),
    admittedHead: shaValue(value.admittedHead, `${path}.admittedHead`, true),
    requiredScopeIds: nullableStringArray(value.requiredScopeIds, `${path}.requiredScopeIds`),
    eligible: booleanValue(value.eligible, `${path}.eligible`),
    ...(Object.prototype.hasOwnProperty.call(value, 'reviewedSource')
      ? { reviewedSource: validateReviewedSourceBinding(value.reviewedSource, `${path}.reviewedSource`) }
      : {})
  }
}

export function validateRolloutPlan(value: unknown): RolloutPlan {
  if (!isRecord(value)) fail('plan', 'expected object')
  exactKeys(value, [
    'target', 'waves', 'concurrency', 'promotionPolicy', 'rows', 'retryOf', 'exclusions',
    ...(Object.prototype.hasOwnProperty.call(value, 'inventoryRevision') ? ['inventoryRevision'] : [])
  ], 'plan')
  if (!Array.isArray(value.waves) || value.waves.length === 0) fail('plan.waves', 'expected non-empty array')
  if (!Array.isArray(value.rows)) fail('plan.rows', 'expected array')
  if (value.rows.length === 0 || value.rows.length > MAX_ROLLOUT_INSTALLATIONS) {
    fail('plan.rows', 'row count is outside release bounds')
  }

  const waves = value.waves.map((wave, waveIndex) => stringArray(wave, `plan.waves[${waveIndex}]`, true))
  const rows = value.rows.map((row, index) => validatePlanRow(row, `plan.rows[${index}]`))
  if (waves.length > MAX_ROLLOUT_INSTALLATIONS || waves.some(wave => wave.length === 0)) {
    fail('plan.waves', 'waves must be non-empty and within release bounds')
  }
  const members = waves.flat()
  if (members.length === 0 || members.length > MAX_ROLLOUT_INSTALLATIONS) {
    fail('plan.waves', 'wave count is outside release bounds')
  }
  if (new Set(members).size !== members.length) fail('plan.waves', 'installation appears in multiple waves')
  if (new Set(rows.map(row => row.installId)).size !== rows.length) fail('plan.rows', 'duplicate installation')
  if (!members.every(installId => rows.some(row => row.installId === installId)))
    fail('plan.waves', 'wave contains unknown installation')
  const rowsById = new Map(rows.map(row => [row.installId, row]))
  if (members.some(installId => !rowsById.get(installId)?.eligible)) {
    fail('plan.waves', 'ineligible installation is assigned to a wave')
  }
  const exclusions = stringArray(value.exclusions, 'plan.exclusions', true)
  if (exclusions.some(installId => !rows.some(row => row.installId === installId)))
    fail('plan.exclusions', 'unknown excluded installation')
  if (exclusions.some(installId => members.includes(installId)))
    fail('plan.exclusions', 'excluded installation is still assigned to a wave')
  if (new Set([...members, ...exclusions]).size !== rows.length)
    fail('plan', 'every row must be assigned to one wave or exclusion')

  return {
    target: validateRolloutTarget(value.target),
    ...(Object.prototype.hasOwnProperty.call(value, 'inventoryRevision')
      ? { inventoryRevision: stringValue(value.inventoryRevision, 'plan.inventoryRevision') } : {}),
    waves,
    concurrency: (() => {
      const concurrency = integerValue(value.concurrency, 'plan.concurrency', 1)
      if (concurrency > MAX_ROLLOUT_CONCURRENCY) fail('plan.concurrency', 'concurrency exceeds release bounds')
      return concurrency
    })(),
    promotionPolicy: enumValue(value.promotionPolicy, PROMOTION_POLICIES, 'plan.promotionPolicy'),
    rows,
    retryOf: nullableString(value.retryOf, 'plan.retryOf'),
    exclusions
  }
}

export function validateRolloutCommand(value: unknown): RolloutCommand {
  if (!isRecord(value)) fail('command', 'expected object')
  exactKeys(
    value,
    ['id', 'expectedRevision', 'requestId', 'action', 'installId', 'reason', 'promotionPolicy'],
    'command'
  )

  const action = enumValue(value.action, ROLLOUT_ACTIONS, 'command.action')
  const targeted = new Set<RolloutAction>(['reprobe', 'recover', 'exclude'])
  const installId = value.installId === null ? null : installIdValue(value.installId, 'command.installId')
  const reason = value.reason === null ? null : stringValue(value.reason, 'command.reason')
  const promotionPolicy =
    value.promotionPolicy === null
      ? null
      : enumValue(value.promotionPolicy, PROMOTION_POLICIES, 'command.promotionPolicy')

  if (targeted.has(action) && !installId) fail('command.installId', 'targeted action requires installation')
  if (!targeted.has(action) && installId) fail('command.installId', 'action does not accept installation')
  if (action !== 'exclude' && action !== 'archive' && reason !== null)
    fail('command.reason', 'action does not accept reason')
  if (action === 'exclude' && !reason) fail('command.reason', 'exclude requires reason')
  if (action === 'set-policy' && !promotionPolicy) fail('command.promotionPolicy', 'set-policy requires policy')
  if (action !== 'set-policy' && promotionPolicy !== null)
    fail('command.promotionPolicy', 'action does not accept policy')

  return {
    id: stringValue(value.id, 'command.id'),
    expectedRevision: integerValue(value.expectedRevision, 'command.expectedRevision'),
    requestId: stringValue(value.requestId, 'command.requestId'),
    action,
    installId,
    reason,
    promotionPolicy
  }
}

export function assertRolloutSnapshot(value: unknown): asserts value is RolloutSnapshot {
  validateRolloutSnapshot(value)
}

export function isRolloutSnapshot(value: unknown): value is RolloutSnapshot {
  try {
    validateRolloutSnapshot(value)
    return true
  } catch {
    return false
  }
}
