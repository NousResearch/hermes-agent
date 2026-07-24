// Cross-window transcript freshness. Each desktop window owns its own gateway
// transport for a session, so a turn that completes in window A never streams
// into window B. This bus pings peers with a lightweight (sessionId, count)
// signature so a stale window can re-pull history before the user sends into
// an out-of-date context (#65047). Distinct from hermes:sessions (sidebar list).

export type TranscriptChangedPayload = {
  /** Stored session id shared across Desktop windows. */
  sessionId: string
  /** ChatMessage-normalized length (after toChatMessages), not raw SessionMessage count. */
  messageCount: number
  updatedAt?: number
}

const CHANNEL = 'hermes:transcript'

const channel = typeof BroadcastChannel === 'undefined' ? null : new BroadcastChannel(CHANNEL)

export function broadcastTranscriptChanged(payload: TranscriptChangedPayload): void {
  if (!payload.sessionId) {
    return
  }

  channel?.postMessage(payload)
}

export function onTranscriptChanged(handler: (payload: TranscriptChangedPayload) => void): () => void {
  if (!channel) {
    return () => {}
  }

  const listener = (event: MessageEvent<TranscriptChangedPayload>) => {
    const data = event.data

    if (!data || typeof data.sessionId !== 'string' || typeof data.messageCount !== 'number') {
      return
    }

    handler(data)
  }

  channel.addEventListener('message', listener)

  return () => channel.removeEventListener('message', listener)
}
