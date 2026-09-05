import type { AgentNoticePayload } from '@/store/agent-notices'

import type { CompletionEvent } from '../../plugins/kanban/completion-notify'
import type { KanbanTask } from '../../plugins/kanban/types'

export type WorldSource = 'agent_notice' | 'gateway' | 'kanban' | 'pull_request' | 'system'
export type WorldSeverity = 'info' | 'success' | 'warning' | 'error' | 'critical'
export type WorldScope = 'worker' | 'task' | 'building' | 'district' | 'city'

export type WorldActionKind =
  | 'inspect'
  | 'inspect_blocker'
  | 'comment'
  | 'recover_task'
  | 'reassign_task'
  | 'reclaim_task'
  | 'create_task'
  | 'create_session'
  | 'dialogue_send'
  | 'voice_send'
  | 'approval_response'
  | 'request_approval'
  | 'show_source'

export interface WorldSourceRef {
  board?: string
  taskId?: string
  agentId?: string
  prId?: string
}

export interface ExternalWorldEventInput {
  source: 'gateway' | 'pull_request' | 'system'
  id: string
  kind: string
  occurredAt?: number
  title: string
  detail?: string
  severity?: WorldSeverity
  scope?: WorldScope
  sourceRef?: WorldSourceRef
  facts?: Record<string, unknown>
}

export interface PullRequestWorldEventInput {
  author?: string
  base?: string
  head?: string
  id: string
  number?: number
  repository?: string
  status:
    | 'approved'
    | 'checks_failed'
    | 'closed'
    | 'conflict'
    | 'draft_merged'
    | 'merged'
    | 'review_findings'
    | 'review_requested'
  title: string
  url?: string
}

export interface ReleaseWorldEventInput {
  id: string
  name: string
  version?: string
  status: 'failed' | 'started' | 'succeeded'
  detail?: string
}

export interface GatewayWorldEventInput {
  id: string
  profile?: string
  status: 'auth_failed' | 'connected' | 'degraded' | 'disconnected' | 'failed'
  detail?: string
}

export interface ApprovalWorldEventInput {
  actionId?: string
  id: string
  sessionId?: string
  status: 'granted' | 'rejected' | 'required'
  title?: string
}

export interface CreditWorldEventInput {
  id: string
  remainingPercent?: number
  status: 'depleted' | 'low' | 'reset'
}

export interface ProfileLifecycleWorldEventInput {
  id: string
  profile: string
  role?: string
  status: 'active' | 'blocked' | 'crashed' | 'idle' | 'working'
}

export interface WorldEvent {
  id: string
  source: WorldSource
  kind: string
  occurredAt: number
  receivedAt: number
  severity: WorldSeverity
  scope: WorldScope
  sourceRef?: WorldSourceRef
  title: string
  detail?: string
  facts: Record<string, unknown>
  actionKinds: WorldActionKind[]
  transition: true
}

export interface WorldCondition {
  id: string
  source: WorldSource
  kind: string
  severity: WorldSeverity
  scope: WorldScope
  sourceRef?: WorldSourceRef
  title: string
  detail?: string
  facts: Record<string, unknown>
  active: true
}

const WORKER_STALE_SECONDS = 120

function text(value: unknown): string {
  return typeof value === 'string' ? value.trim() : ''
}

function finiteNumber(value: unknown, fallback: number): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback
}

function record(value: unknown): Record<string, unknown> {
  return value && typeof value === 'object' && !Array.isArray(value) ? { ...(value as Record<string, unknown>) } : {}
}

function safeDetail(value: unknown): string | undefined {
  const detail = text(value)

  return detail ? detail.slice(0, 2_000) : undefined
}

function sourceEventId(source: WorldSource, id: unknown, fallback: string): string {
  const normalized = typeof id === 'number' && Number.isFinite(id) ? String(id) : text(id)

  return normalized ? `${source}:${normalized}` : `${source}:${fallback}`
}

function eventSpec(kind: string): {
  actionKinds: WorldActionKind[]
  scope: WorldScope
  severity: WorldSeverity
  worldKind: string
} {
  switch (kind) {
    case 'blocked':
      return {
        actionKinds: ['inspect_blocker', 'comment', 'reassign_task', 'reclaim_task'],
        scope: 'task',
        severity: 'warning',
        worldKind: 'task.blocked'
      }

    case 'block_loop_detected':
      return {
        actionKinds: ['inspect_blocker', 'comment', 'reassign_task'],
        scope: 'district',
        severity: 'critical',
        worldKind: 'task.block_loop'
      }

    case 'crashed':
      return {
        actionKinds: ['inspect', 'reclaim_task', 'reassign_task'],
        scope: 'worker',
        severity: 'error',
        worldKind: 'worker.crashed'
      }

    case 'gave_up':
      return {
        actionKinds: ['inspect', 'reclaim_task', 'reassign_task'],
        scope: 'task',
        severity: 'error',
        worldKind: 'worker.gave_up'
      }

    case 'timed_out':
      return {
        actionKinds: ['inspect', 'reclaim_task', 'reassign_task'],
        scope: 'task',
        severity: 'warning',
        worldKind: 'worker.timed_out'
      }

    case 'completed':
      return {
        actionKinds: ['inspect', 'show_source'],
        scope: 'task',
        severity: 'success',
        worldKind: 'task.completed'
      }

    case 'done':
      return {
        actionKinds: ['inspect', 'show_source'],
        scope: 'task',
        severity: 'success',
        worldKind: 'task.completed'
      }

    case 'unblocked':
      return {
        actionKinds: ['inspect', 'show_source'],
        scope: 'task',
        severity: 'success',
        worldKind: 'task.recovered'
      }

    case 'archived':

    case 'deleted':
      return { actionKinds: ['inspect'], scope: 'task', severity: 'info', worldKind: 'task.removed' }

    case 'review':
      return { actionKinds: ['inspect', 'comment'], scope: 'task', severity: 'warning', worldKind: 'task.in_review' }

    case 'running':
      return { actionKinds: ['inspect'], scope: 'task', severity: 'info', worldKind: 'task.running' }

    case 'ready':
      return { actionKinds: ['inspect'], scope: 'task', severity: 'info', worldKind: 'task.ready' }

    case 'scheduled':

    case 'todo':

    case 'triage':
      return { actionKinds: ['inspect'], scope: 'task', severity: 'info', worldKind: 'task.waiting' }

    default:
      return {
        actionKinds: ['inspect', 'show_source'],
        scope: 'city',
        severity: 'warning',
        worldKind: 'system.unclassified_alert'
      }
  }
}

function eventTitle(kind: string, payload: Record<string, unknown>, fallback: string): string {
  return text(payload.title) || text(payload.message) || fallback || kind.replace(/_/g, ' ')
}

/** Convert a Kanban event into a stable, presentation-independent world event. */
export function normalizeKanbanEvent(board: string, event: CompletionEvent, now = Date.now()): WorldEvent {
  const kind = text(event.kind) || 'unknown'
  const payload = record(event.payload)
  const taskId = text(event.task_id)
  const spec = eventSpec(kind)
  const fallback = taskId ? `Kanban task ${taskId}` : `Kanban ${kind}`
  const reason = safeDetail(payload.reason) || safeDetail(payload.error) || safeDetail(payload.summary)

  return {
    id: sourceEventId('kanban', event.id, `${board}:${taskId}:${kind}`),
    source: 'kanban',
    kind: spec.worldKind,
    occurredAt: finiteNumber(payload.created_at, now),
    receivedAt: now,
    severity: spec.severity,
    scope: spec.scope,
    sourceRef: { board, ...(taskId ? { taskId } : {}) },
    title: eventTitle(kind, payload, fallback),
    ...(reason ? { detail: reason } : {}),
    facts: { board, taskId, backendKind: kind, ...payload },
    actionKinds: spec.actionKinds,
    transition: true
  }
}

/** Convert a Hermes notification into a world event without changing toast behavior. */
export function normalizeAgentNotice(payload: AgentNoticePayload, now = Date.now()): WorldEvent | null {
  const detail = safeDetail(payload.text)

  if (!detail) {
    return null
  }

  const level = text(payload.level)
  const key = text(payload.key) || text(payload.id) || detail

  const kind =
    key === 'credits.depleted'
      ? 'credits.depleted'
      : level === 'success'
        ? 'agent.success'
        : level === 'error'
          ? 'agent.error'
          : level === 'warn'
            ? 'agent.warning'
            : 'agent.info'

  const severity: WorldSeverity =
    level === 'success' ? 'success' : level === 'error' ? 'error' : level === 'warn' ? 'warning' : 'info'

  return {
    id: sourceEventId('agent_notice', key, detail),
    source: 'agent_notice',
    kind,
    occurredAt: now,
    receivedAt: now,
    severity,
    scope: kind === 'credits.depleted' ? 'city' : 'building',
    title: kind.replace('.', ' '),
    detail,
    facts: { key, level, noticeKind: text(payload.kind) },
    actionKinds: ['inspect'],
    transition: true
  }
}

/** Normalize future PR/release/gateway inputs while retaining their source identity. */
export function normalizeExternalEvent(input: ExternalWorldEventInput, now = Date.now()): WorldEvent {
  const kind = text(input.kind) || 'unclassified_alert'
  const severity = input.severity ?? 'warning'
  const scope = input.scope ?? 'city'
  const detail = safeDetail(input.detail)

  return {
    id: sourceEventId(input.source, input.id, `${kind}:${input.title}`),
    source: input.source,
    kind: kind.includes('.') ? kind : `system.${kind}`,
    occurredAt: finiteNumber(input.occurredAt, now),
    receivedAt: now,
    severity,
    scope,
    ...(input.sourceRef ? { sourceRef: input.sourceRef } : {}),
    title: text(input.title) || kind.replace(/_/g, ' '),
    ...(detail ? { detail } : {}),
    facts: { ...(input.facts ?? {}), sourceKind: kind },
    actionKinds: ['inspect', 'show_source'],
    transition: true
  }
}

export function normalizePullRequestEvent(input: PullRequestWorldEventInput, now = Date.now()): WorldEvent {
  const kindByStatus: Record<PullRequestWorldEventInput['status'], string> = {
    approved: 'pr.approved',
    checks_failed: 'pr.review_findings',
    closed: 'pr.closed',
    conflict: 'pr.merge_conflict',
    draft_merged: 'pr.merged_draft',
    merged: 'pr.merged_stable',
    review_findings: 'pr.review_findings',
    review_requested: 'pr.review_requested'
  }

  const severityByStatus: Record<PullRequestWorldEventInput['status'], WorldSeverity> = {
    approved: 'success',
    checks_failed: 'error',
    closed: 'info',
    conflict: 'warning',
    draft_merged: 'success',
    merged: 'success',
    review_findings: 'warning',
    review_requested: 'info'
  }

  return normalizeExternalEvent(
    {
      facts: {
        author: input.author,
        base: input.base,
        head: input.head,
        number: input.number,
        repository: input.repository,
        status: input.status,
        url: input.url
      },
      id: input.id,
      kind: kindByStatus[input.status],
      scope: input.status === 'merged' ? 'city' : 'district',
      severity: severityByStatus[input.status],
      source: 'pull_request',
      sourceRef: { prId: input.id },
      title: input.title
    },
    now
  )
}

export function normalizeReleaseEvent(input: ReleaseWorldEventInput, now = Date.now()): WorldEvent {
  const kindByStatus: Record<ReleaseWorldEventInput['status'], string> = {
    failed: 'release.failed',
    started: 'release.started',
    succeeded: 'release.succeeded'
  }

  return normalizeExternalEvent(
    {
      detail: input.detail,
      facts: { status: input.status, version: input.version },
      id: input.id,
      kind: kindByStatus[input.status],
      scope: input.status === 'started' ? 'district' : 'city',
      severity: input.status === 'failed' ? 'critical' : input.status === 'succeeded' ? 'success' : 'info',
      source: 'system',
      title: input.name
    },
    now
  )
}

export function normalizeGatewayEvent(input: GatewayWorldEventInput, now = Date.now()): WorldEvent {
  const kindByStatus: Record<GatewayWorldEventInput['status'], string> = {
    auth_failed: 'auth.failed',
    connected: 'gateway.connected',
    degraded: 'gateway.degraded',
    disconnected: 'gateway.disconnected',
    failed: 'gateway.disconnected'
  }

  return normalizeExternalEvent(
    {
      detail: input.detail,
      facts: { profile: input.profile, status: input.status },
      id: input.id,
      kind: kindByStatus[input.status],
      scope: input.status === 'connected' ? 'building' : 'city',
      severity:
        input.status === 'connected'
          ? 'success'
          : input.status === 'degraded'
            ? 'warning'
            : input.status === 'auth_failed'
              ? 'error'
              : 'critical',
      source: 'gateway',
      title: input.profile ? `Gateway ${input.profile}` : 'Gateway'
    },
    now
  )
}

export function normalizeApprovalEvent(input: ApprovalWorldEventInput, now = Date.now()): WorldEvent {
  const kindByStatus: Record<ApprovalWorldEventInput['status'], string> = {
    granted: 'approval.granted',
    rejected: 'approval.rejected',
    required: 'approval.required'
  }

  return normalizeExternalEvent(
    {
      facts: { actionId: input.actionId, sessionId: input.sessionId, status: input.status },
      id: input.id,
      kind: kindByStatus[input.status],
      scope: 'task',
      severity: input.status === 'granted' ? 'success' : input.status === 'rejected' ? 'warning' : 'info',
      source: 'system',
      title: input.title || 'Approval checkpoint'
    },
    now
  )
}

export function normalizeCreditEvent(input: CreditWorldEventInput, now = Date.now()): WorldEvent {
  const kindByStatus: Record<CreditWorldEventInput['status'], string> = {
    depleted: 'credits.depleted',
    low: 'credits.low',
    reset: 'credits.reset'
  }

  return normalizeExternalEvent(
    {
      facts: { remainingPercent: input.remainingPercent, status: input.status },
      id: input.id,
      kind: kindByStatus[input.status],
      scope: input.status === 'reset' ? 'district' : 'city',
      severity: input.status === 'depleted' ? 'critical' : input.status === 'low' ? 'warning' : 'success',
      source: 'system',
      title: input.status === 'reset' ? 'Credits restored' : 'Credit capacity'
    },
    now
  )
}

export function normalizeProfileLifecycleEvent(input: ProfileLifecycleWorldEventInput, now = Date.now()): WorldEvent {
  const kindByStatus: Record<ProfileLifecycleWorldEventInput['status'], string> = {
    active: 'agent.active',
    blocked: 'task.blocked',
    crashed: 'worker.crashed',
    idle: 'agent.idle',
    working: 'task.running'
  }

  return normalizeExternalEvent(
    {
      facts: { profile: input.profile, role: input.role, status: input.status },
      id: input.id,
      kind: kindByStatus[input.status],
      scope: input.status === 'crashed' ? 'worker' : 'building',
      severity: input.status === 'blocked' ? 'warning' : input.status === 'crashed' ? 'error' : 'info',
      source: 'system',
      sourceRef: { agentId: input.profile },
      title: input.profile
    },
    now
  )
}

/** Derive a persistent world condition from the current Kanban card snapshot. */
export function classifyTaskCondition(task: KanbanTask, now = Date.now()): WorldCondition | null {
  const status = text(task.status).toLowerCase()

  if (!status) {
    return null
  }

  const spec = eventSpec(status)

  const stale =
    status === 'running' && !!task.last_heartbeat_at && now / 1_000 - task.last_heartbeat_at > WORKER_STALE_SECONDS

  const kind = stale ? 'worker.stale' : spec.worldKind
  const severity: WorldSeverity = stale ? 'warning' : spec.severity
  const scope: WorldScope = stale ? 'worker' : spec.scope
  const title = task.title.trim() || `Kanban task ${task.id}`
  const detail = stale ? 'Worker heartbeat missing for more than 2 minutes.' : task.latest_summary?.trim() || undefined

  return {
    id: `kanban:condition:${task.id}`,
    source: 'kanban',
    kind,
    severity,
    scope,
    sourceRef: { taskId: task.id, ...(task.assignee ? { agentId: task.assignee } : {}) },
    title,
    ...(detail ? { detail } : {}),
    facts: {
      status,
      assignee: task.assignee ?? null,
      priority: task.priority ?? null,
      workerPid: task.worker_pid ?? null,
      lastHeartbeatAt: task.last_heartbeat_at ?? null,
      stale
    },
    active: true
  }
}

export function worldEventId(event: Pick<WorldEvent, 'source' | 'id'>): string {
  return `${event.source}:${event.id}`
}

/** Merge a new frame into the visible feed without replaying a transition. */
export function dedupeWorldEvents(previous: readonly WorldEvent[], incoming: readonly WorldEvent[]): WorldEvent[] {
  const byId = new Map(previous.map(event => [worldEventId(event), event]))

  for (const event of incoming) {
    byId.set(worldEventId(event), event)
  }

  return [...byId.values()].sort((left, right) => left.occurredAt - right.occurredAt)
}
