import type { StatusbarMenuItem } from '@/app/shell/statusbar-controls'

const LOG_TAIL = 5

interface RpcEventLike {
  payload?: unknown
  session_id?: string
  type?: string
}

function asRecord(payload: unknown): Record<string, unknown> {
  return payload && typeof payload === 'object' ? (payload as Record<string, unknown>) : {}
}

/**
 * Return true only for a current-session compression boundary request.
 *
 * The gateway can finish a turn after the desktop has already moved to a
 * different chat. A late boundary must not create a fresh draft in whichever
 * session happens to be focused when it arrives.
 */
export function shouldRequestFreshSessionAfterCompression(
  event: RpcEventLike,
  activeSessionId: null | string
): boolean {
  if (event.type !== 'dashboard.new_session_requested') {
    return false
  }

  if (asRecord(event.payload).reason !== 'compression_exhausted') {
    return false
  }

  if (!activeSessionId) {
    return false
  }

  return !event.session_id ? true : event.session_id === activeSessionId
}

/** Return the bounded, redacted handoff prompt carried by a current boundary. */
export function compressionBoundaryResumePrompt(
  event: RpcEventLike,
  activeSessionId: null | string
): string | null {
  if (!shouldRequestFreshSessionAfterCompression(event, activeSessionId)) {
    return null
  }

  const prompt = asRecord(event.payload).resume_prompt

  return typeof prompt === 'string' && prompt.trim() ? prompt.trim() : null
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

export function approvalReplaySessionId(
  eventType: string | undefined,
  activeSessionId: null | string,
  routedSessionId: null | string
): null | string {
  if (eventType === 'gateway.ready') {
    return activeSessionId
  }

  if (eventType === 'session.info') {
    return routedSessionId
  }

  return null
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
