const MAX_TRACKED_SESSIONS = 256
const RECENT_SEQUENCE_WINDOW = 1024

interface SessionSequenceState {
  highest: number
  recent: Set<number>
}

interface DedupableGatewayEvent {
  connectionId?: string
  payload?: unknown
  replayEpoch?: string
  seq?: number
  session_id?: string
  type?: string
}

/**
 * Deduplicates sequenced backend events after Desktop socket fan-in.
 *
 * A backend process gives each session event a sequence number. Multiple
 * renderer sockets can receive the same frame, so the renderer—not an
 * individual socket—owns this bounded record of delivered frames.
 */
export class GatewayEventDeduper {
  private readonly sessions = new Map<string, SessionSequenceState>()

  /**
   * Returns whether an event should reach renderer stores.
   *
   * Seq-less and session-less events deliberately pass: legacy backends and
   * global notifications have no ordering identity to deduplicate safely.
   */
  accept(event: DedupableGatewayEvent): boolean {
    this.forgetReclaimedSession(event)

    const sessionId = event.session_id
    const seq = event.seq

    if (!sessionId || typeof seq !== 'number' || !Number.isFinite(seq)) {
      return true
    }

    const key = `${this.epochKey(event)}\u0000${sessionId}`
    let state = this.sessions.get(key)

    if (!state || state.highest - seq > RECENT_SEQUENCE_WINDOW) {
      state = { highest: seq, recent: new Set([seq]) }
      this.touch(key, state)

      return true
    }

    if (state.recent.has(seq)) {
      this.touch(key, state)

      return false
    }

    state.highest = Math.max(state.highest, seq)
    state.recent.add(seq)

    while (state.recent.size > RECENT_SEQUENCE_WINDOW) {
      const oldest = state.recent.values().next().value

      if (oldest === undefined) {
        break
      }

      state.recent.delete(oldest)
    }

    this.touch(key, state)

    return true
  }

  private epochKey(event: DedupableGatewayEvent): string {
    if (typeof event.replayEpoch === 'string' && event.replayEpoch) {
      return event.replayEpoch
    }

    return `unknown:${event.connectionId ?? 'local'}`
  }

  private forgetReclaimedSession(event: DedupableGatewayEvent): void {
    if (event.type !== 'session.reclaimed') {
      return
    }

    const sessionId = (event.payload as { session_id?: unknown } | undefined)?.session_id

    if (typeof sessionId !== 'string' || !sessionId) {
      return
    }

    for (const key of this.sessions.keys()) {
      if (key.endsWith(`\u0000${sessionId}`)) {
        this.sessions.delete(key)
      }
    }
  }

  private touch(key: string, state: SessionSequenceState): void {
    this.sessions.delete(key)
    this.sessions.set(key, state)

    while (this.sessions.size > MAX_TRACKED_SESSIONS) {
      const oldest = this.sessions.keys().next().value

      if (oldest === undefined) {
        return
      }

      this.sessions.delete(oldest)
    }
  }
}
