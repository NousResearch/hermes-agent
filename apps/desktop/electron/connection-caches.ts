/**
 * Main-process caches keyed by connection id.
 *
 * Every entry here answers "what is at connection <id>", so it is only valid while that id keeps
 * pointing at the same machine. Two lifecycle events break that: removing a connection (ids are
 * recycled label slugs — `connectionIdForLabel` suffixes only against CURRENTLY taken ids, so
 * re-adding "Mac mini" gets `mac-mini` back) and editing its dial material (same id, new host).
 * Neither used to evict, and `shouldRetrySshInventory` never retries a cached success, so the old
 * machine's profile list stayed authoritative for the rest of the app session.
 *
 * They live in one module so that invariant has somewhere to be stated — and tested — instead of
 * being three loose `Map`s in main.ts.
 */

/** Last known profile list per ssh connection (reused so switching back to local doesn't empty Bot Mode). */
export const sshRosterCache = new Map<string, string[]>()

/** When the ssh inventory probe last ran, for the retry backoff. */
export const sshInventoryAttemptedAt = new Map<string, number>()

/**
 * Stable backend identity per connection (the `install_id` its /api/status reports; absent on
 * older backends). TTL-cached because enumeration runs on the ~5s Bot Mode roster poll.
 */
export const connectionInstallIds = new Map<string, { id?: string; ts: number }>()

const CONNECTION_SCOPED_CACHES: Map<string, unknown>[] = [sshRosterCache, sshInventoryAttemptedAt, connectionInstallIds]

/**
 * Forget everything cached about a connection id. Call whenever that id stops naming the machine
 * it named — it is removed, or its dial material changes — so the next probe re-learns from the
 * live target instead of serving the previous one.
 */
export function evictConnectionCaches(connectionId: string): void {
  const id = String(connectionId || '')

  if (!id) {
    return
  }

  for (const cache of CONNECTION_SCOPED_CACHES) {
    cache.delete(id)
  }
}
