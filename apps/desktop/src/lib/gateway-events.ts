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
 * Unscoped stream events that must stay pinned to the session that received
 * ``message.start`` after the user switches chats mid-turn (#47709 / #48281).
 * Without this, ``explicitSid || activeSessionId`` reattributes live deltas to
 * the newly focused chat.
 */
/** Unscoped stream events that must stay pinned to the session that received
 * ``message.start`` after the user switches chats mid-turn (#47709 / #48281).
 * Without this, ``explicitSid || activeSessionId`` reattributes live deltas to
 * the newly focused chat. Exported so the event handler can tell which events
 * are pin-eligible when deciding whether an unpinned straggler is legitimate. */
export const UNSCOPED_STREAM_EVENT_TYPES = new Set([
  'approval.request',
  'browser.progress',
  'clarify.request',
  'error',
  'mcp.setup.request',
  'message.complete',
  'message.delta',
  'message.interim',
  'message.start',
  'reasoning.available',
  'reasoning.delta',
  'secret.request',
  'status.update',
  'sudo.request',
  'thinking.delta',
  'tool.complete',
  'tool.generating',
  'tool.progress',
  'tool.start'
])

const UNSCOPED_STREAM_END_EVENT_TYPES = new Set(['error', 'message.complete'])

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

export interface GatewayEventSessionRouteInput {
  activeSessionId: null | string
  eventType: string | undefined
  explicitSessionId: string
  unscopedStreamSessionId: null | string
}

export interface GatewayEventSessionRoute {
  drop: boolean
  nextUnscopedStreamSessionId: null | string
  /** True when the event was attributed via the pinned stream session rather
   *  than the active-session fallback. The caller uses this to drop late
   *  stragglers: an unpinned stream event landing on a session that has no
   *  live turn belongs to a turn that already ended elsewhere. */
  pinned: boolean
  sessionId: null | string
}

/** Which session (if any) to re-pull `approval.pending` for after `eventType`.
 *
 *  `gateway.ready` and `session.info` are the two rehydration points. An
 *  UNSCOPED `session.info` (the approvals-loop / broadcast fan-out, no
 *  `session_id` on the frame) reaches here attributed to the active session by
 *  the routing fallback; when `isGone(activeSessionId)` — the gateway already
 *  answered 4001 for that runtime — replaying would only re-send the dead id
 *  on every fan-out tick (#100639), so return null. A frame that names the
 *  session explicitly is the runtime speaking for itself and is never gone. */
export function approvalReplaySessionId(
  eventType: string | undefined,
  activeSessionId: null | string,
  routedSessionId: null | string,
  options?: { explicit?: boolean; isGone?: (sessionId: string) => boolean }
): null | string {
  let target: null | string = null

  if (eventType === 'gateway.ready') {
    target = activeSessionId
  } else if (eventType === 'session.info') {
    target = routedSessionId
  }

  if (target && !options?.explicit && options?.isGone?.(target)) {
    return null
  }

  return target
}

/**
 * Resolve which runtime session owns a gateway event.
 *
 * Explicit ``session_id`` always wins. Unscoped stream events pin to the
 * session that received ``message.start`` so a mid-turn chat switch cannot
 * steal live deltas / tool events onto the newly focused transcript.
 */
export function resolveGatewayEventSessionId({
  activeSessionId,
  eventType,
  explicitSessionId,
  unscopedStreamSessionId
}: GatewayEventSessionRouteInput): GatewayEventSessionRoute {
  if (explicitSessionId) {
    const nextUnscopedStreamSessionId =
      eventType && UNSCOPED_STREAM_END_EVENT_TYPES.has(eventType) && explicitSessionId === unscopedStreamSessionId
        ? null
        : unscopedStreamSessionId

    return {
      drop: false,
      nextUnscopedStreamSessionId,
      pinned: true,
      sessionId: explicitSessionId
    }
  }

  if (gatewayEventRequiresSessionId(eventType)) {
    return {
      drop: true,
      nextUnscopedStreamSessionId: unscopedStreamSessionId,
      pinned: false,
      sessionId: null
    }
  }

  const streamEvent = eventType ? UNSCOPED_STREAM_EVENT_TYPES.has(eventType) : false

  const sessionId =
    eventType === 'message.start'
      ? activeSessionId
      : streamEvent
        ? unscopedStreamSessionId || activeSessionId
        : activeSessionId

  let nextUnscopedStreamSessionId = unscopedStreamSessionId

  if (eventType === 'message.start' && activeSessionId) {
    nextUnscopedStreamSessionId = activeSessionId
  } else if (eventType && UNSCOPED_STREAM_END_EVENT_TYPES.has(eventType)) {
    nextUnscopedStreamSessionId = null
  }

  return {
    drop: false,
    nextUnscopedStreamSessionId,
    pinned: streamEvent && eventType !== 'message.start' && Boolean(unscopedStreamSessionId),
    sessionId
  }
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
