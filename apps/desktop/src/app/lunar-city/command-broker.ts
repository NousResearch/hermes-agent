import { entityKey as canonicalEntityKey } from './identity'
import type { LeaderOwner } from './leader-sessions'
import type {
  AuthorityState,
  DestinationId,
  EntityIdentity,
  EntityKey,
  LunarCitySnapshot,
  LunarEntity,
  SourceHealth
} from './model'

export type CommandVerification = 'verified' | 'rejected' | 'timed_out' | 'verification_required'

export type CommandOperation =
  | 'open-session'
  | 'inspect-evidence'
  | 'send-guidance'
  | 'interrupt-session'
  | 'interrupt-subagent'
  | 'terminate-run'
  | 'retry-task'
  | 'reclaim-task'
  | 'reassign-task'
  | 'dispatch-task'
  | 'change-task-state'

export type CommandEvidenceKind = 'attachments' | 'comments' | 'diagnostics' | 'events' | 'log' | 'run' | 'task'

export type CommandIntent =
  | { kind: 'open-session'; entityKey: EntityKey }
  | { kind: 'inspect-evidence'; entityKey: EntityKey; evidence: CommandEvidenceKind }
  | { kind: 'send-guidance'; entityKey: EntityKey; text: string }
  | { kind: 'interrupt-session'; entityKey: EntityKey }
  | { kind: 'interrupt-subagent'; entityKey: EntityKey }
  | { kind: 'terminate-run'; entityKey: EntityKey }
  | { kind: 'retry-task'; entityKey: EntityKey }
  | { kind: 'reclaim-task'; entityKey: EntityKey }
  | { kind: 'reassign-task'; entityKey: EntityKey; assignee: string }
  | { kind: 'dispatch-task'; entityKey: EntityKey }
  | { kind: 'change-task-state'; entityKey: EntityKey; state: string }

export type CommandEffectKind =
  | 'session-present'
  | 'evidence-present'
  | 'guidance-recorded'
  | 'session-interrupted'
  | 'subagent-interrupted'
  | 'run-terminated'
  | 'task-retried'
  | 'task-reclaimed'
  | 'task-reassigned'
  | 'task-dispatched'
  | 'task-state-changed'

export interface CommandEffect {
  kind: CommandEffectKind
  targetId: string
  value?: string
}

export interface ReadbackPlan {
  expectedEffect: CommandEffect
  id: string
  kind: 'session' | 'subagent' | 'kanban-task' | 'kanban-run'
}

export interface CommandObservedState {
  animation: string
  authority: AuthorityState
  destination: DestinationId
  observedAt: number
  source: string
  value: string
}

export interface CommandPlanContext {
  canonicalProjectId?: string
  currentState: string
  repositoryId?: string
  source: SourceHealth
  sourceOwner: LeaderOwner
}

export interface CommandPlan {
  confirmation: boolean
  consequence: string
  context: CommandPlanContext
  digest: string
  entityKey: EntityKey
  identity: EntityIdentity
  intent: CommandIntent
  method: string
  operation: CommandOperation
  owner: LeaderOwner
  params: Readonly<Record<string, unknown>>
  plannedAt: number
  plannedRevision: number
  readback: ReadbackPlan
}

export interface CommandReceipt {
  error?: string
  identity: EntityIdentity
  readback?: CommandReadback | null
  response?: unknown
  verification: CommandVerification
}

export interface CommandTargetState {
  availableOperations: readonly CommandOperation[]
  canonicalProjectId?: string
  entity: LunarEntity
  observedState: CommandObservedState
  ownerCandidates: readonly LeaderOwner[]
  readbackCapabilities: readonly ReadbackPlan['kind'][]
  repositoryId?: string
  source: SourceHealth
  sourceOwner: LeaderOwner
}

export interface CommandPlanningSnapshot {
  city: LunarCitySnapshot
  targets: ReadonlyMap<EntityKey, CommandTargetState>
}

export interface CommandCausalReceipt {
  authority: 'authoritative'
  planDigest: string
}

export interface CommandCurrentAuthority {
  authority: 'authoritative'
  identity: EntityIdentity
  observedAt: number
  owner: LeaderOwner
}

export interface CommandReadback {
  authority: AuthorityState
  effect?: CommandEffect
  identity: EntityIdentity
  observedAt: number
  operation: CommandOperation
  outcome: 'rejected' | 'unknown' | 'verified'
  owner: LeaderOwner
  receipt?: CommandCausalReceipt
  revision: number
}

export interface CommandExecutor {
  /** Performs one exact-owner source read immediately before a mutation. */
  currentAuthority?: (plan: CommandPlan) => Promise<CommandCurrentAuthority | null>
  readback(plan: CommandPlan): Promise<CommandReadback | null>
  send(plan: CommandPlan): Promise<unknown>
}

export interface CommandExecutors {
  kanbanRun: CommandExecutor
  kanbanTask: CommandExecutor
  session: CommandExecutor
  subagent: CommandExecutor
}

export interface ExecuteCommandOptions {
  confirmed?: boolean
  latestSnapshot: () => CommandPlanningSnapshot | Promise<CommandPlanningSnapshot>
}

export type CommandPlanRevalidation = { canonicalPlan: CommandPlan; ok: true } | { error: string; ok: false }

export class CommandRejectedError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'CommandRejectedError'
  }
}

export class CommandTimeoutError extends Error {
  constructor(
    message: string,
    readonly possiblyApplied = true
  ) {
    super(message)
    this.name = 'CommandTimeoutError'
  }
}

interface Compatibility {
  confirmation: boolean
  consequence: string
  method: string
  params: Record<string, unknown>
  readback: ReadbackPlan
}

function fail(code: string, detail: string): never {
  throw new Error(`${code}: ${detail}`)
}

function required(value: string | undefined, code: string, label: string): string {
  const normalized = value?.trim() ?? ''

  if (!normalized) {
    fail(code, `${label} is required`)
  }

  return normalized
}

function validateIdentity(identity: EntityIdentity): void {
  required(identity.connectionId, 'identity-incomplete', 'connectionId')
  required(identity.profile, 'identity-incomplete', 'profile')

  if (identity.kind === 'session' || identity.kind === 'subagent') {
    required(identity.sessionId, 'identity-incomplete', 'sessionId')
  }

  if (identity.kind === 'subagent') {
    required(identity.subagentId, 'identity-incomplete', 'subagentId')
  }

  if (identity.kind === 'kanban') {
    required(identity.board, 'identity-incomplete', 'board')
    required(identity.taskId, 'identity-incomplete', 'taskId')
  }
}

function sameOwner(left: LeaderOwner, right: LeaderOwner): boolean {
  return left.connectionId === right.connectionId && left.profile === right.profile
}

function sameIdentity(left: EntityIdentity, right: EntityIdentity): boolean {
  if (left.kind !== right.kind || left.connectionId !== right.connectionId || left.profile !== right.profile) {
    return false
  }

  if (left.kind === 'profile' && right.kind === 'profile') {
    return true
  }

  if (left.kind === 'session' && right.kind === 'session') {
    return left.sessionId === right.sessionId
  }

  if (left.kind === 'subagent' && right.kind === 'subagent') {
    return left.sessionId === right.sessionId && left.subagentId === right.subagentId
  }

  if (left.kind === 'kanban' && right.kind === 'kanban') {
    return (
      left.board === right.board &&
      left.taskId === right.taskId &&
      left.runId === right.runId &&
      left.workerId === right.workerId
    )
  }

  return false
}

function samePosition(left: LunarEntity['position'], right: LunarEntity['position']): boolean {
  if (!left || !right) {
    return left === right
  }

  return left.x === right.x && left.y === right.y && left.z === right.z
}

function sameEntity(left: LunarEntity, right: LunarEntity): boolean {
  return (
    left.key === right.key &&
    sameIdentity(left.identity, right.identity) &&
    left.authority === right.authority &&
    left.observedAt === right.observedAt &&
    left.destination === right.destination &&
    left.animation === right.animation &&
    left.sourceState === right.sourceState &&
    left.projectId === right.projectId &&
    left.variant === right.variant &&
    samePosition(left.position, right.position)
  )
}

function exactOwner(identity: EntityIdentity, target: CommandTargetState): LeaderOwner {
  if (target.ownerCandidates.length !== 1) {
    fail('owner-is-ambiguous', `expected one owner route and found ${target.ownerCandidates.length}`)
  }

  const candidate = target.ownerCandidates[0]

  const owner = {
    connectionId: required(candidate.connectionId, 'owner-is-ambiguous', 'connectionId'),
    profile: required(candidate.profile, 'owner-is-ambiguous', 'profile')
  }

  if (
    owner.connectionId !== identity.connectionId ||
    owner.profile !== identity.profile ||
    !sameOwner(owner, target.sourceOwner)
  ) {
    fail('owner-is-ambiguous', 'the resolved route and source owner do not own the complete typed identity')
  }

  return owner
}

function exactSource(target: CommandTargetState, city: LunarCitySnapshot): SourceHealth {
  const sources = city.sources.filter(source => source.source === target.source.source)

  if (sources.length !== 1) {
    fail('source-is-ambiguous', `expected one source named ${target.source.source}`)
  }

  const source = sources[0]

  if (
    source.authority !== target.source.authority ||
    source.observedAt !== target.source.observedAt ||
    source.error !== target.source.error
  ) {
    fail('target-binding-mismatch', 'target source differs from the immutable city source')
  }

  return { ...source }
}

function validateTargetBinding(
  intent: CommandIntent,
  target: CommandTargetState,
  cityEntity: LunarEntity,
  source: SourceHealth
): void {
  if (
    target.entity.key !== intent.entityKey ||
    canonicalEntityKey(target.entity.identity) !== intent.entityKey ||
    !sameEntity(target.entity, cityEntity)
  ) {
    fail('target-binding-mismatch', 'target entity differs from the immutable city entity')
  }

  const state = target.observedState

  if (
    state.animation !== cityEntity.animation ||
    state.authority !== cityEntity.authority ||
    state.destination !== cityEntity.destination ||
    state.observedAt !== cityEntity.observedAt ||
    state.source !== source.source ||
    source.observedAt !== cityEntity.observedAt ||
    source.authority !== cityEntity.authority ||
    !state.value.trim()
  ) {
    fail('target-binding-mismatch', 'current state evidence is not bound to the immutable city observation')
  }
}

function requireKind(
  identity: EntityIdentity,
  operation: CommandOperation,
  allowed: readonly EntityIdentity['kind'][]
): void {
  if (!allowed.includes(identity.kind)) {
    fail('operation-identity-incompatible', `${operation} cannot target ${identity.kind}`)
  }
}

function identityParams(identity: EntityIdentity): Record<string, unknown> {
  if (identity.kind === 'session') {
    return { session_id: identity.sessionId }
  }

  if (identity.kind === 'subagent') {
    return { session_id: identity.sessionId, subagent_id: identity.subagentId }
  }

  if (identity.kind === 'kanban') {
    return {
      board: identity.board,
      task_id: identity.taskId,
      ...(identity.runId ? { run_id: identity.runId } : {}),
      ...(identity.workerId ? { worker_id: identity.workerId } : {})
    }
  }

  return {}
}

function effect(kind: CommandEffectKind, targetId: string, value?: string): CommandEffect {
  return { kind, targetId, ...(value === undefined ? {} : { value }) }
}

function readback(kind: ReadbackPlan['kind'], id: string, expectedEffect: CommandEffect): ReadbackPlan {
  return { expectedEffect, id, kind }
}

function compatibilityFor(intent: CommandIntent, identity: EntityIdentity, currentState: string): Compatibility {
  const params = identityParams(identity)

  switch (intent.kind) {
    case 'open-session': {
      requireKind(identity, intent.kind, ['session', 'subagent'])
      const sessionId = identity.kind === 'session' || identity.kind === 'subagent' ? identity.sessionId : ''

      return {
        confirmation: false,
        consequence: 'Open the exact owning standard session without changing execution.',
        method: 'session.open',
        params,
        readback: readback('session', sessionId, effect('session-present', sessionId))
      }
    }

    case 'inspect-evidence': {
      requireKind(identity, intent.kind, ['session', 'subagent', 'kanban'])

      if (identity.kind === 'session') {
        return {
          confirmation: false,
          consequence: `Inspect ${intent.evidence} from its labeled owning source without changing execution.`,
          method: 'evidence.inspect',
          params: { ...params, evidence: intent.evidence },
          readback: readback(
            'session',
            identity.sessionId,
            effect('evidence-present', identity.sessionId, intent.evidence)
          )
        }
      }

      if (identity.kind === 'subagent') {
        return {
          confirmation: false,
          consequence: `Inspect ${intent.evidence} from its labeled owning source without changing execution.`,
          method: 'evidence.inspect',
          params: { ...params, evidence: intent.evidence },
          readback: readback(
            'subagent',
            identity.subagentId,
            effect('evidence-present', identity.subagentId, intent.evidence)
          )
        }
      }

      if (identity.kind !== 'kanban') {
        fail('operation-identity-incompatible', `${intent.kind} cannot target ${identity.kind}`)
      }

      const useRun = intent.evidence === 'run' && Boolean(identity.runId)
      const id = useRun ? identity.runId! : identity.taskId

      return {
        confirmation: false,
        consequence: `Inspect ${intent.evidence} from its labeled owning source without changing execution.`,
        method: 'evidence.inspect',
        params: { ...params, evidence: intent.evidence },
        readback: readback(useRun ? 'kanban-run' : 'kanban-task', id, effect('evidence-present', id, intent.evidence))
      }
    }

    case 'send-guidance': {
      requireKind(identity, intent.kind, ['session', 'subagent'])
      const text = required(intent.text, 'identity-incomplete', 'guidance text')

      if (identity.kind === 'subagent') {
        return {
          confirmation: false,
          consequence: 'Send ordinary guidance to the exact child worker; it does not assert task completion.',
          method: 'subagent.steer',
          params: { ...params, text },
          readback: readback('subagent', identity.subagentId, effect('guidance-recorded', identity.subagentId))
        }
      }

      if (identity.kind !== 'session') {
        fail('operation-identity-incompatible', `${intent.kind} cannot target ${identity.kind}`)
      }

      return {
        confirmation: false,
        consequence: 'Send ordinary guidance to the exact owned conversation; it does not assert task completion.',
        method: currentState === 'running' ? 'session.steer' : 'prompt.submit',
        params: { ...params, text },
        readback: readback('session', identity.sessionId, effect('guidance-recorded', identity.sessionId))
      }
    }

    case 'interrupt-session': {
      requireKind(identity, intent.kind, ['session'])
      const sessionId = identity.kind === 'session' ? identity.sessionId : ''

      return {
        confirmation: true,
        consequence: 'Interrupt the active turn in the exact owning session; unfinished work may stop.',
        method: 'session.interrupt',
        params,
        readback: readback('session', sessionId, effect('session-interrupted', sessionId))
      }
    }

    case 'interrupt-subagent': {
      requireKind(identity, intent.kind, ['subagent'])
      const subagentId = identity.kind === 'subagent' ? identity.subagentId : ''

      return {
        confirmation: true,
        consequence: 'Interrupt the exact child worker; unfinished delegated work may stop.',
        method: 'subagent.interrupt',
        params,
        readback: readback('subagent', subagentId, effect('subagent-interrupted', subagentId))
      }
    }

    case 'terminate-run': {
      requireKind(identity, intent.kind, ['kanban'])
      const runId = identity.kind === 'kanban' ? required(identity.runId, 'identity-incomplete', 'runId') : ''

      return {
        confirmation: true,
        consequence: 'Terminate the exact Kanban run; the worker may stop before producing acceptance evidence.',
        method: 'kanban.run.terminate',
        params,
        readback: readback('kanban-run', runId, effect('run-terminated', runId))
      }
    }

    case 'retry-task': {
      requireKind(identity, intent.kind, ['kanban'])
      const taskId = identity.kind === 'kanban' ? identity.taskId : ''

      return {
        confirmation: true,
        consequence:
          'Request one fresh retry for the exact task after confirmation; no retry occurs on an ambiguous result.',
        method: 'kanban.task.retry',
        params,
        readback: readback('kanban-task', taskId, effect('task-retried', taskId))
      }
    }

    case 'reclaim-task': {
      requireKind(identity, intent.kind, ['kanban'])
      const taskId = identity.kind === 'kanban' ? identity.taskId : ''

      return {
        confirmation: true,
        consequence: 'Reclaim the exact task from its current worker and return it to dispatcher control.',
        method: 'kanban.task.reclaim',
        params,
        readback: readback('kanban-task', taskId, effect('task-reclaimed', taskId))
      }
    }

    case 'reassign-task': {
      requireKind(identity, intent.kind, ['kanban'])
      const taskId = identity.kind === 'kanban' ? identity.taskId : ''
      const assignee = required(intent.assignee, 'identity-incomplete', 'assignee')

      return {
        confirmation: true,
        consequence: `Reassign the exact task to ${assignee}; its current worker may be replaced.`,
        method: 'kanban.task.reassign',
        params: { ...params, assignee },
        readback: readback('kanban-task', taskId, effect('task-reassigned', taskId, assignee))
      }
    }

    case 'dispatch-task': {
      requireKind(identity, intent.kind, ['kanban'])
      const taskId = identity.kind === 'kanban' ? identity.taskId : ''

      return {
        confirmation: true,
        consequence: 'Dispatch the exact task once; an ambiguous acknowledgement will not be retried.',
        method: 'kanban.task.dispatch',
        params,
        readback: readback('kanban-task', taskId, effect('task-dispatched', taskId))
      }
    }

    case 'change-task-state': {
      requireKind(identity, intent.kind, ['kanban'])
      const taskId = identity.kind === 'kanban' ? identity.taskId : ''
      const state = required(intent.state, 'identity-incomplete', 'task state')

      return {
        confirmation: true,
        consequence: `Change the exact task state from ${currentState} to ${state}.`,
        method: 'kanban.task.patch',
        params: { ...params, status: state },
        readback: readback('kanban-task', taskId, effect('task-state-changed', taskId, state))
      }
    }
  }
}

function mutationRequiresAuthoritativeState(operation: CommandOperation): boolean {
  return operation !== 'open-session' && operation !== 'inspect-evidence'
}

function stableValue(value: unknown): string {
  if (value === undefined) {
    return 'undefined'
  }

  if (value === null || typeof value === 'boolean' || typeof value === 'number' || typeof value === 'string') {
    return JSON.stringify(value)
  }

  if (Array.isArray(value)) {
    return `[${value.map(stableValue).join(',')}]`
  }

  if (typeof value === 'object') {
    const record = value as Record<string, unknown>

    return `{${Object.keys(record)
      .sort()
      .map(key => `${JSON.stringify(key)}:${stableValue(record[key])}`)
      .join(',')}}`
  }

  return JSON.stringify(String(value))
}

function digest(value: unknown): string {
  const input = stableValue(value)
  let hash = 0xcbf29ce484222325n

  for (let index = 0; index < input.length; index += 1) {
    hash ^= BigInt(input.charCodeAt(index))
    hash = BigInt.asUintN(64, hash * 0x100000001b3n)
  }

  return hash.toString(16).padStart(16, '0')
}

function planPayload(plan: Omit<CommandPlan, 'digest'> | CommandPlan): Omit<CommandPlan, 'digest'> {
  const { digest: _digest, ...payload } = plan as CommandPlan

  return payload
}

function cloneAndDeepFreeze<T>(value: T, ancestors = new WeakSet<object>()): T {
  if (value === null || value === undefined || ['boolean', 'number', 'string'].includes(typeof value)) {
    return value
  }

  if (typeof value !== 'object') {
    fail('invalid-command-plan', `unsupported canonical value type ${typeof value}`)
  }

  const objectValue = value as object

  if (ancestors.has(objectValue)) {
    fail('invalid-command-plan', 'cyclic canonical values are not allowed')
  }

  ancestors.add(objectValue)

  try {
    if (Array.isArray(value)) {
      return Object.freeze(value.map(item => cloneAndDeepFreeze(item, ancestors))) as T
    }

    const prototype = Object.getPrototypeOf(objectValue)

    if (prototype !== Object.prototype && prototype !== null) {
      fail('invalid-command-plan', 'canonical values must contain only plain objects and primitives')
    }

    const clone: Record<string, unknown> = {}

    for (const key of Object.keys(objectValue).sort()) {
      clone[key] = cloneAndDeepFreeze((value as Record<string, unknown>)[key], ancestors)
    }

    return Object.freeze(clone) as T
  } finally {
    ancestors.delete(objectValue)
  }
}

function buildPlan(
  intent: CommandIntent,
  identity: EntityIdentity,
  owner: LeaderOwner,
  source: SourceHealth,
  target: CommandTargetState,
  snapshot: CommandPlanningSnapshot,
  compatibility: Compatibility
): CommandPlan {
  const payload = cloneAndDeepFreeze<Omit<CommandPlan, 'digest'>>({
    confirmation: compatibility.confirmation,
    consequence: compatibility.consequence,
    context: {
      canonicalProjectId: target.canonicalProjectId,
      currentState: target.observedState.value,
      repositoryId: target.repositoryId,
      source,
      sourceOwner: target.sourceOwner
    },
    entityKey: intent.entityKey,
    identity,
    intent,
    method: compatibility.method,
    operation: intent.kind,
    owner,
    params: compatibility.params,
    plannedAt: Math.max(target.entity.observedAt, source.observedAt),
    plannedRevision: snapshot.city.revision,
    readback: compatibility.readback
  })

  return cloneAndDeepFreeze({ ...payload, digest: digest(payload) })
}

export function planCommand(intent: CommandIntent, snapshot: CommandPlanningSnapshot): CommandPlan {
  const target = snapshot.targets.get(intent.entityKey)
  const cityEntity = snapshot.city.entities.get(intent.entityKey)

  if (!target || !cityEntity) {
    fail('target-not-found', 'the immutable city snapshot does not contain the requested entity')
  }

  validateIdentity(target.entity.identity)

  const owner = exactOwner(target.entity.identity, target)
  const source = exactSource(target, snapshot.city)
  validateTargetBinding(intent, target, cityEntity, source)

  const compatibility = compatibilityFor(intent, target.entity.identity, target.observedState.value)

  if (!target.availableOperations.includes(intent.kind)) {
    fail('unsupported-command', `${intent.kind} is not available for this exact target`)
  }

  if (!target.readbackCapabilities.includes(compatibility.readback.kind)) {
    fail('readback-unavailable', `${compatibility.readback.kind} cannot be authoritatively reread on this owner route`)
  }

  if (mutationRequiresAuthoritativeState(intent.kind)) {
    if (target.entity.authority === 'stale' || source.authority === 'stale') {
      fail('target-is-stale', 'refresh exact authoritative state before staging a write')
    }

    if (target.entity.authority === 'partial' || source.authority === 'partial') {
      fail('target-is-partial', 'partial state cannot authorize a write')
    }

    if (target.entity.authority !== 'authoritative' || source.authority !== 'authoritative') {
      fail('target-is-unavailable', 'unknown state cannot authorize a write')
    }
  }

  return buildPlan(intent, target.entity.identity, owner, source, target, snapshot, compatibility)
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

/** Reads each caller-owned plan field once into an isolated, deeply frozen value. */
function snapshotCallerPlan(plan: CommandPlan): CommandPlan {
  return cloneAndDeepFreeze({
    confirmation: plan.confirmation,
    consequence: plan.consequence,
    context: plan.context,
    digest: plan.digest,
    entityKey: plan.entityKey,
    identity: plan.identity,
    intent: plan.intent,
    method: plan.method,
    operation: plan.operation,
    owner: plan.owner,
    params: plan.params,
    plannedAt: plan.plannedAt,
    plannedRevision: plan.plannedRevision,
    readback: plan.readback
  })
}

function revalidateSnapshotPlan(received: CommandPlan, latest: CommandPlanningSnapshot): CommandPlanRevalidation {
  if (digest(planPayload(received)) !== received.digest) {
    return { error: 'Invalid command plan: canonical digest mismatch.', ok: false }
  }

  let rebuilt: CommandPlan

  try {
    rebuilt = planCommand(received.intent, latest)
  } catch (error) {
    return { error: `target-changed-since-plan: ${errorMessage(error)}`, ok: false }
  }

  if (stableValue(received) !== stableValue(rebuilt)) {
    return {
      error: 'target-changed-since-plan: canonical plan fields no longer match the latest target.',
      ok: false
    }
  }

  return { canonicalPlan: rebuilt, ok: true }
}

/** Rebuilds and returns a newly isolated canonical plan from the latest target. */
export function revalidateCommandPlan(plan: CommandPlan, latest: CommandPlanningSnapshot): CommandPlanRevalidation {
  try {
    return revalidateSnapshotPlan(snapshotCallerPlan(plan), latest)
  } catch (error) {
    return { error: `Invalid command plan: ${errorMessage(error)}`, ok: false }
  }
}

export function commandPlanIntegrityError(plan: CommandPlan, latest: CommandPlanningSnapshot): string | undefined {
  const result = revalidateCommandPlan(plan, latest)

  return result.ok ? undefined : result.error
}

function executorFor(plan: CommandPlan, executors: CommandExecutors): CommandExecutor {
  switch (plan.readback.kind) {
    case 'session':
      return executors.session

    case 'subagent':
      return executors.subagent

    case 'kanban-task':
      return executors.kanbanTask

    case 'kanban-run':
      return executors.kanbanRun
  }
}

function receipt(
  plan: CommandPlan,
  verification: CommandVerification,
  extras: Omit<CommandReceipt, 'identity' | 'verification'> = {}
): CommandReceipt {
  return { identity: { ...plan.identity }, verification, ...extras }
}

function authoritativeCausalReadback(plan: CommandPlan, value: CommandReadback): boolean {
  const monotonic = value.revision > plan.plannedRevision && value.observedAt > plan.plannedAt
  const receiptMatches = value.receipt?.authority === 'authoritative' && value.receipt.planDigest === plan.digest

  return Boolean(monotonic || receiptMatches)
}

function readbackBaseMatches(plan: CommandPlan, value: CommandReadback): boolean {
  return (
    value.authority === 'authoritative' &&
    value.operation === plan.operation &&
    sameIdentity(value.identity, plan.identity) &&
    sameOwner(value.owner, plan.owner) &&
    authoritativeCausalReadback(plan, value)
  )
}

function currentAuthorityMatches(plan: CommandPlan, value: CommandCurrentAuthority): boolean {
  return (
    value.authority === 'authoritative' &&
    sameIdentity(value.identity, plan.identity) &&
    sameOwner(value.owner, plan.owner) &&
    value.observedAt >= plan.plannedAt
  )
}

function sameEffect(left: CommandEffect | undefined, right: CommandEffect): boolean {
  return left?.kind === right.kind && left.targetId === right.targetId && left.value === right.value
}

/**
 * Revalidates against a freshly injected immutable target immediately before
 * one send, then requires a causal operation-specific authoritative readback.
 */
export async function executeCommand(
  plan: CommandPlan,
  executors: CommandExecutors,
  options: ExecuteCommandOptions
): Promise<CommandReceipt> {
  let receivedPlan: CommandPlan

  try {
    receivedPlan = snapshotCallerPlan(plan)
  } catch (error) {
    return {
      error: `Invalid command plan: ${errorMessage(error)}`,
      identity: { connectionId: 'invalid-command-plan', kind: 'profile', profile: 'invalid-command-plan' },
      verification: 'rejected'
    }
  }

  if (receivedPlan.confirmation && options.confirmed !== true) {
    return receipt(receivedPlan, 'rejected', { error: 'Command confirmation was not granted.' })
  }

  if (!options?.latestSnapshot) {
    return receipt(receivedPlan, 'rejected', { error: 'Invalid command plan: latest target revalidation is required.' })
  }

  let latest: CommandPlanningSnapshot

  try {
    latest = await options.latestSnapshot()
  } catch (error) {
    return receipt(receivedPlan, 'rejected', { error: `Latest target revalidation failed: ${errorMessage(error)}` })
  }

  const validation = revalidateSnapshotPlan(receivedPlan, latest)

  if (!validation.ok) {
    return receipt(receivedPlan, 'rejected', { error: validation.error })
  }

  const canonicalPlan = validation.canonicalPlan
  const executor = executorFor(canonicalPlan, executors)

  if (mutationRequiresAuthoritativeState(canonicalPlan.operation)) {
    if (!executor.currentAuthority) {
      return receipt(canonicalPlan, 'rejected', {
        error: 'Exact current authority reader is unavailable; nothing was sent.'
      })
    }

    let authority: CommandCurrentAuthority | null

    try {
      authority = await executor.currentAuthority(canonicalPlan)
    } catch (error) {
      return receipt(canonicalPlan, 'rejected', {
        error: `Exact current authority read failed before send: ${errorMessage(error)}`
      })
    }

    if (!authority || !currentAuthorityMatches(canonicalPlan, authority)) {
      return receipt(canonicalPlan, 'rejected', {
        error: 'Exact current authority was unavailable, stale, cached, or owned by another route; nothing was sent.'
      })
    }
  }

  let response: unknown

  try {
    response = await executor.send(canonicalPlan)
  } catch (error) {
    if (error instanceof CommandRejectedError) {
      return receipt(canonicalPlan, 'rejected', { error: error.message })
    }

    if (error instanceof CommandTimeoutError && !error.possiblyApplied) {
      return receipt(canonicalPlan, 'timed_out', { error: error.message })
    }

    return receipt(canonicalPlan, 'verification_required', { error: errorMessage(error) })
  }

  let readbackValue: CommandReadback | null

  try {
    readbackValue = await executor.readback(canonicalPlan)
  } catch (error) {
    return receipt(canonicalPlan, 'verification_required', { error: errorMessage(error), response })
  }

  if (!readbackValue || !readbackBaseMatches(canonicalPlan, readbackValue)) {
    return receipt(canonicalPlan, 'verification_required', {
      error: 'Authoritative causal readback was missing, cached, stale, or mismatched.',
      readback: readbackValue,
      response
    })
  }

  // A canonical authoritative rejection is terminal even when the requested
  // effect is absent or differs; rejection means the effect did not occur.
  if (readbackValue.outcome === 'rejected') {
    return receipt(canonicalPlan, 'rejected', { readback: readbackValue, response })
  }

  if (
    readbackValue.outcome !== 'verified' ||
    !sameEffect(readbackValue.effect, canonicalPlan.readback.expectedEffect)
  ) {
    return receipt(canonicalPlan, 'verification_required', {
      error: 'Authoritative readback did not contain the expected canonical effect.',
      readback: readbackValue,
      response
    })
  }

  return receipt(canonicalPlan, 'verified', { readback: readbackValue, response })
}
