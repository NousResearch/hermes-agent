/**
 * Main-process memo: which remote profile owns a session id when the caller gave
 * no profile hint (#85834).
 *
 * #58485: the memo only ever INSERTS — the TTL marks entries stale but never
 * evicts — so a long-lived main process kept one entry per session id it had
 * ever resolved. That is unbounded growth on the main-process heap (the
 * renderer-side leak fixes never touched it). Entries are FIFO-evicted past
 * `limit`, the same shape as titleCache's bound.
 */

export interface RemoteOwnerEntry {
  at: number
  profile: null | string
}

export interface RemoteOwnerCache {
  /** Fresh memo for `sessionId`, else null (absent or older than `ttlMs`). */
  fresh(sessionId: string): null | RemoteOwnerEntry
  /** Record a lookup result, FIFO-evicting the oldest entry past `limit`. */
  remember(sessionId: string, profile: null | string): void
  /** Test/inspection hook: current size. */
  size(): number
}

export function createRemoteOwnerCache({
  limit = 500,
  now = () => Date.now(),
  ttlMs = 30_000
}: { limit?: number; now?: () => number; ttlMs?: number } = {}): RemoteOwnerCache {
  const entries = new Map<string, RemoteOwnerEntry>()

  return {
    fresh(sessionId) {
      const cached = entries.get(sessionId)

      return cached && now() - cached.at < ttlMs ? cached : null
    },
    remember(sessionId, profile) {
      if (entries.size >= limit) {
        entries.delete(entries.keys().next().value)
      }

      entries.set(sessionId, { at: now(), profile })
    },
    size: () => entries.size
  }
}
