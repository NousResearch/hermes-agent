import type { StatusbarMenuItem } from '@/app/shell/statusbar-controls'

const LOG_TAIL = 5

interface RpcEventLike {
  payload?: unknown
  type?: string
}

function asRecord(payload: unknown): Record<string, unknown> {
  return payload && typeof payload === 'object' ? (payload as Record<string, unknown>) : {}
}

/**
 * Whether an unscoped event (no `session_id`) must be dropped rather than
 * attributed to the focused chat.
 *
 * Desktop multiplexes concurrent sessions and windows over one event pipeline.
 * Any transcript-mutating or prompt-blocking event without an explicit
 * `session_id` is ambiguous there, so attributing it to the focused chat leaks
 * turns across sessions. The backend now stamps these per-session events
 * explicitly, so an unscoped copy is malformed and must be dropped.
 *
 * Keep genuinely global UI events unscoped (`gateway.ready`, skin changes,
 * preview restart progress, startup `session.info` without a live session).
 */
export function gatewayEventRequiresSessionId(eventType: string | undefined): boolean {
  if (!eventType) {
    return false
  }

  return (
    eventType.startsWith('subagent.') ||
    eventType.startsWith('message.') ||
    eventType.startsWith('reasoning.') ||
    eventType.startsWith('tool.') ||
    eventType === 'clarify.request' ||
    eventType === 'approval.request' ||
    eventType === 'sudo.request' ||
    eventType === 'secret.request' ||
    eventType === 'status.update' ||
    eventType === 'review.summary'
  )
}

export function gatewayEventCompletedFileDiff(event: RpcEventLike): boolean {
  if (event.type !== 'tool.complete') {
    return false
  }

  const diff = asRecord(event.payload).inline_diff

  return typeof diff === 'string' && diff.trim().length > 0
}

export function buildGatewayLogItems(lines: readonly string[]): readonly StatusbarMenuItem[] {
  if (lines.length === 0) {
    return [
      {
        className: 'text-muted-foreground',
        disabled: true,
        id: 'gateway-log-empty',
        label: 'No recent gateway log lines'
      }
    ]
  }

  return lines.slice(-LOG_TAIL).map((line, index) => ({
    className: 'font-mono text-[0.68rem] text-muted-foreground',
    disabled: true,
    id: `gateway-log:${index}`,
    label: line.trim().slice(0, 120) || '(blank log line)'
  }))
}
