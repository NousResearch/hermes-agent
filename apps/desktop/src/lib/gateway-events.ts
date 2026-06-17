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
 * Only `subagent.*` qualifies: it describes background/async work that must
 * never attach to whichever chat happens to be focused. Every other scoped
 * event — message/reasoning/thinking/tool/status/prompt — is, when unscoped,
 * the active turn's own output. The gateway always stamps a *background*
 * session's events with that session's id, so a missing id can only mean "the
 * focused turn". #42178 dropped those too, which silently swallowed the live
 * answer; it then reappeared only after a transcript refetch (manual refresh).
 */
export function gatewayEventRequiresSessionId(eventType: string | undefined): boolean {
  return eventType?.startsWith('subagent.') ?? false
}

export interface GatewaySessionBusyState {
  awaitingResponse?: boolean
  busy?: boolean
}

/**
 * Resolve which runtime session owns an unscoped gateway event.
 *
 * Background sessions are stamped with their id server-side; a missing id
 * usually means the focused turn. When multiple sessions are still busy
 * (e.g. user opened a new chat while the previous turn streams), prefer the
 * active session if it is busy, otherwise the sole busy session, and drop
 * ambiguous unscoped events rather than attributing them to a quiet session.
 */
export function resolveGatewayEventSessionId(
  explicitSid: string | undefined | null,
  activeSessionId: string | null,
  sessionStates: ReadonlyMap<string, GatewaySessionBusyState>
): string | null {
  const normalizedExplicit = (explicitSid || '').trim()

  if (normalizedExplicit) {
    return normalizedExplicit
  }

  const busySessions: string[] = []

  for (const [runtimeId, state] of sessionStates.entries()) {
    if (state.busy || state.awaitingResponse) {
      busySessions.push(runtimeId)
    }
  }

  if (busySessions.length === 1) {
    return busySessions[0]
  }

  if (busySessions.length > 1) {
    if (activeSessionId && busySessions.includes(activeSessionId)) {
      return activeSessionId
    }

    return null
  }

  return activeSessionId
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
