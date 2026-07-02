export type SessionStatusKind =
  | 'blocked_by_monitor'
  | 'compression_tip'
  | 'model_policy_violation'
  | 'model_request_waiting'
  | 'queued_steer'
  | 'queued_steer_blocked'
  | 'running_tool'
  | 'stale_db_runtime_active'
  | 'terminal_recovery'

export interface SessionStatusLabel {
  kind: SessionStatusKind
  label: string
  title: string
  tone: 'destructive' | 'muted' | 'success' | 'warning'
}

interface SessionStatusEvidence {
  _lineage_root_id?: null | string
  compression_tip_session_id?: null | string
  id?: null | string
  is_active?: boolean
  last_tool_runtime_event?: unknown
  lineage_root?: null | string
  model_policy_recommended_action?: null | string
  model_policy_violation?: boolean
  model_request_high_context?: boolean
  model_request_queued_steer_count?: null | number | string
  model_request_status?: null | string
  model_request_steer_queued?: boolean
  queued_steer_count?: null | number | string
  required_model?: null | string
  status_evidence_source?: null | readonly string[]
  steer_boundary?: null | string
  terminal_recovery_needed?: boolean
}

const MONITOR_BLOCKED_SOURCES = new Set(['blocked', 'blocked_by_monitor', 'monitor_blocked'])
const SAFE_LABEL_TOKEN_RE = /^[A-Za-z0-9_.:/-]{1,120}$/
const TOOL_BOUNDARY_STEER = 'cannot_steer_until_current_tool_boundary'

function hasSource(sources: null | readonly string[] | undefined, source: string): boolean {
  return Array.isArray(sources) && sources.includes(source)
}

function hasMonitorBlockedSource(sources: null | readonly string[] | undefined): boolean {
  return Array.isArray(sources) && sources.some(source => MONITOR_BLOCKED_SOURCES.has(source))
}

function safePositiveCount(value: null | number | string | undefined): number {
  const count = typeof value === 'string' ? Number(value) : value

  return Number.isFinite(count) && Number(count) > 0 ? Number(count) : 0
}

function safeLabelToken(value: null | string | undefined): string {
  const token = String(value ?? '').trim()

  return SAFE_LABEL_TOKEN_RE.test(token) ? token : ''
}

export function getSessionStatusLabels(session: SessionStatusEvidence): SessionStatusLabel[] {
  const labels: SessionStatusLabel[] = []
  const sources = session.status_evidence_source

  const queuedSteerCount = Math.max(
    safePositiveCount(session.queued_steer_count),
    safePositiveCount(session.model_request_queued_steer_count)
  )

  const modelRequestStatus = safeLabelToken(session.model_request_status)

  const terminalRecoveryNeeded =
    session.terminal_recovery_needed === true ||
    modelRequestStatus === 'terminal_recovery_needed' ||
    hasSource(sources, 'terminal_recovery_needed')

  const steerBlocked =
    queuedSteerCount > 0 &&
    (session.steer_boundary === TOOL_BOUNDARY_STEER ||
      session.model_request_steer_queued === true ||
      (modelRequestStatus === 'waiting' && session.model_request_high_context === true))

  if (session.model_policy_violation === true) {
    const requiredModel = safeLabelToken(session.required_model)

    labels.push({
      kind: 'model_policy_violation',
      label: 'Model policy',
      title: requiredModel
        ? `Fixed-model policy requires ${requiredModel}.`
        : 'A fixed-model policy violation is attached to this session.',
      tone: 'destructive'
    })
  }

  if (hasMonitorBlockedSource(sources)) {
    labels.push({
      kind: 'blocked_by_monitor',
      label: 'Blocked by monitor evidence',
      title: 'A monitor-safe blocker signal is attached to this session.',
      tone: 'destructive'
    })
  }

  if (session.last_tool_runtime_event != null) {
    labels.push({
      kind: 'running_tool',
      label: 'Running tool',
      title: 'A value-free runtime tool event is attached to this session.',
      tone: 'success'
    })
  }

  if (terminalRecoveryNeeded) {
    labels.push({
      kind: 'terminal_recovery',
      label: 'Terminal recovery',
      title: 'Terminal recovery is needed; inspect DB, logs, repo state, and resume compactly.',
      tone: 'warning'
    })
  }

  if (modelRequestStatus === 'waiting' && session.model_request_high_context === true) {
    labels.push({
      kind: 'model_request_waiting',
      label: 'High-context wait',
      title: 'A high-context model request is waiting for the backend or next tool boundary.',
      tone: 'warning'
    })
  }

  if (steerBlocked) {
    labels.push({
      kind: 'queued_steer_blocked',
      label: 'Steer waiting on tool',
      title: `${queuedSteerCount} queued steer${queuedSteerCount === 1 ? '' : 's'} waiting for the active tool boundary without exposing prompt text.`,
      tone: 'warning'
    })
  } else if (queuedSteerCount > 0) {
    labels.push({
      kind: 'queued_steer',
      label: 'Queued steer',
      title: `${queuedSteerCount} queued steer${queuedSteerCount === 1 ? '' : 's'} waiting without exposing prompt text.`,
      tone: 'warning'
    })
  }

  if (
    hasSource(sources, 'compression_projection') ||
    (session.compression_tip_session_id && session.id && session.compression_tip_session_id !== session.id) ||
    (session._lineage_root_id && session._lineage_root_id !== session.id) ||
    (session.lineage_root && session.lineage_root !== session.id)
  ) {
    labels.push({
      kind: 'compression_tip',
      label: 'Compression tip',
      title: 'This row is projected to the live compression tip.',
      tone: 'muted'
    })
  }

  if (hasSource(sources, 'active_session_registry')) {
    labels.push({
      kind: 'stale_db_runtime_active',
      label: 'Stale DB counter, runtime active',
      title: 'Runtime registry evidence says this session is active even if DB counters lag.',
      tone: 'warning'
    })
  }

  return labels
}
