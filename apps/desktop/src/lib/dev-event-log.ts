import type { RpcEvent } from '@/types/hermes'

// Dev-only gateway event tap. Feeds `window.__hermesEventLog` so external
// tooling (apps/desktop/scripts/cdp-drive.mjs `record`) can capture the real
// event stream for transcript-lifecycle debugging (thinking splits, orphaned
// tool rows, vanished cards). No-op in production builds.

interface LoggedEvent {
  seq: number
  ts: number
  type: string
  session_id?: string
  payload?: Record<string, unknown>
}

interface HermesEventLog {
  drain: (afterSeq?: number) => LoggedEvent[]
  size: () => number
}

declare global {
  interface Window {
    __hermesEventLog?: HermesEventLog
  }
}

const MAX_EVENTS = 1000
const TEXT_CAP = 200

let seq = 0
const buffer: LoggedEvent[] = []

/** Compact a payload for logging: cap long strings, summarize the ui card. */
function compactPayload(payload: unknown): Record<string, unknown> | undefined {
  if (!payload || typeof payload !== 'object') {
    return undefined
  }

  const out: Record<string, unknown> = {}

  for (const [key, value] of Object.entries(payload as Record<string, unknown>)) {
    if (key === 'ui' && value && typeof value === 'object') {
      const ui = value as Record<string, unknown>

      out.ui = { uri: ui.uri, htmlLen: typeof ui.html === 'string' ? ui.html.length : 0 }
    } else if (typeof value === 'string' && value.length > TEXT_CAP) {
      out[key] = `${value.slice(0, TEXT_CAP)}…[${value.length}ch]`
    } else {
      out[key] = value
    }
  }

  return out
}

export function tapGatewayEvent(event: RpcEvent): void {
  if (!import.meta.env.DEV) {
    return
  }

  if (!window.__hermesEventLog) {
    window.__hermesEventLog = {
      drain: (afterSeq = 0) => buffer.filter(e => e.seq > afterSeq),
      size: () => buffer.length
    }
  }

  seq += 1
  buffer.push({
    seq,
    ts: Date.now(),
    type: event.type,
    session_id: event.session_id,
    payload: compactPayload(event.payload)
  })

  if (buffer.length > MAX_EVENTS) {
    buffer.splice(0, buffer.length - MAX_EVENTS)
  }
}
