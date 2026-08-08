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
const UNSCOPED_STREAM_EVENT_TYPES = new Set([
  'approval.request',
  'browser.progress',
  'clarify.request',
  'error',
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
  /**
   * Sessions with an unscoped stream in flight, in ``message.start`` order.
   *
   * One pin per concurrent stream. A single shared pin could not represent two
   * chats streaming at once: the second ``message.start`` overwrote the first,
   * and every later unscoped event from the first stream then resolved to the
   * second chat (#46194 / #62823).
   */
  unscopedStreamSessionIds: readonly string[]
}

export interface GatewayEventSessionRoute {
  drop: boolean
  nextUnscopedStreamSessionIds: readonly string[]
  sessionId: null | string
}

const withStreamPin = (pins: readonly string[], sessionId: string): readonly string[] =>
  pins.includes(sessionId) ? pins : [...pins, sessionId]

const withoutStreamPin = (pins: readonly string[], sessionId: string): readonly string[] =>
  pins.includes(sessionId) ? pins.filter(pin => pin !== sessionId) : pins

/**
 * Which in-flight stream owns an unscoped stream event.
 *
 * Returns `null` when ownership is genuinely ambiguous, so the caller drops the
 * event instead of grafting one chat's output onto another.
 */
function resolveUnscopedStreamOwner(pins: readonly string[], activeSessionId: null | string): null | string {
  // No concurrent streams: preserve the established single-stream behaviour —
  // the lone pin owns it, and with no pin at all the focused chat does. The
  // late-event case for that second branch is #70376's subject, not this one.
  if (pins.length <= 1) {
    return pins[0] ?? activeSessionId
  }

  // Two or more streams are live. The gateway stamps background sessions'
  // events with their own id, so an unscoped one is the focused turn's output —
  // but only when the focused chat is itself mid-stream.
  if (activeSessionId && pins.includes(activeSessionId)) {
    return activeSessionId
  }

  // The focused chat is idle, so this belongs to one of several background
  // streams and nothing in the event says which. Guessing is what painted A's
  // deltas onto B; drop instead. The store keeps the correct rows, so the
  // transcript recovers on refetch.
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
  unscopedStreamSessionIds
}: GatewayEventSessionRouteInput): GatewayEventSessionRoute {
  const streamEnd = eventType ? UNSCOPED_STREAM_END_EVENT_TYPES.has(eventType) : false

  if (explicitSessionId) {
    return {
      drop: false,
      // Retire only the pin this event names. Streams still running in other
      // chats keep theirs.
      nextUnscopedStreamSessionIds: streamEnd
        ? withoutStreamPin(unscopedStreamSessionIds, explicitSessionId)
        : unscopedStreamSessionIds,
      sessionId: explicitSessionId
    }
  }

  if (gatewayEventRequiresSessionId(eventType)) {
    return {
      drop: true,
      nextUnscopedStreamSessionIds: unscopedStreamSessionIds,
      sessionId: null
    }
  }

  if (eventType === 'message.start') {
    return {
      drop: false,
      // Add a pin rather than replace one, so a second chat starting a turn
      // cannot take ownership of a stream that is already running elsewhere.
      nextUnscopedStreamSessionIds: activeSessionId
        ? withStreamPin(unscopedStreamSessionIds, activeSessionId)
        : unscopedStreamSessionIds,
      sessionId: activeSessionId
    }
  }

  if (!(eventType && UNSCOPED_STREAM_EVENT_TYPES.has(eventType))) {
    return {
      drop: false,
      nextUnscopedStreamSessionIds: unscopedStreamSessionIds,
      sessionId: activeSessionId
    }
  }

  const owner = resolveUnscopedStreamOwner(unscopedStreamSessionIds, activeSessionId)

  if (!owner) {
    return {
      drop: true,
      nextUnscopedStreamSessionIds: unscopedStreamSessionIds,
      sessionId: null
    }
  }

  return {
    drop: false,
    nextUnscopedStreamSessionIds: streamEnd
      ? withoutStreamPin(unscopedStreamSessionIds, owner)
      : unscopedStreamSessionIds,
    sessionId: owner
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
