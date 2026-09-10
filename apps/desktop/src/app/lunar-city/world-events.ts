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
