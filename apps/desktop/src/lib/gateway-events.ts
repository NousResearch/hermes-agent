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
 * Global broadcasts that carry no session scope are exempt. Everything else
 * - message, reasoning, thinking, tool, status, prompt, session.info,
 * clarify, approval, error - is session-scoped and must carry an explicit
 * session_id.  When the gateway fails to stamp one (race, background-process
 * delivery, subagent mirror, stale transport), dropping the event is safer
 * than silently attributing it to whichever session happens to be focused.
 *
 * Fixes #49106 (Web/WeChat session history leak / stream bleed across
 * sessions) and the #47709 family of desktop stream-bleed bugs.
 */
export function gatewayEventRequiresSessionId(eventType: string | undefined): boolean {
  if (!eventType) {
    return true
  }
  // Truly global broadcasts that are not owned by any single session.
  if (eventType === 'gateway.ready' || eventType === 'skin.changed') {
    return false
  }
  return true
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
