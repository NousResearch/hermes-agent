/**
 * Bounded deferral for session sync writes refused by the runtime-owner gate.
 *
 * The boot-time restore sync (auto-mark-read on reopened tabs, the pin
 * re-assert) re-writes state marks minutes after login, while the gateway
 * that owns the profile is still starting: its control socket does not exist
 * yet, the write cannot be forwarded to the owner, and the offline store's
 * reservation refuses — the owner holds the ownership lock from process
 * start. The backend answers with the named, retryable `owner_unavailable`
 * refusal; deferral is the designed outcome, not an error.
 *
 * The queue holds one retry chain per key. Every retry re-enters the store
 * function that parked the write, and that store's catch decides
 * parked-vs-expired on every re-entry — so no store semantics live here and
 * the queue never touches UI state. The chain can tell its outcomes apart:
 * a still-deferred re-entry throws the marker (after its re-park refreshed
 * this entry and scheduled the next re-entry — the entry STAYS), a landed
 * write resolves (the entry goes — no further re-entry was ever scheduled),
 * and an expired or genuinely failed re-entry rejects with something else
 * (the entry goes; expiry already deleted it in the park call). The bound is
 * finite and cannot be reset by re-creating an entry mid-chain: the counter
 * lives on an entry that only expiry or settlement removes.
 */

/** Retry cadence for a parked write. */
export const OWNER_DEFERRED_RETRY_MS = 30_000

/** Total park-then-retry attempts before a deferral expires. */
export const OWNER_DEFERRED_MAX_ATTEMPTS = 20

/**
 * Thrown to a parked write's caller (and to this module's retry chain) while
 * the deferral stands. Exported for the stores' catch blocks and tests.
 */
export class OwnerWriteDeferredMarker extends Error {
  constructor() {
    super('Session write deferred: runtime owner unavailable')
  }
}

interface DeferredEntry {
  attempts: number
}

const deferred = new Map<string, DeferredEntry>()

function settle(key: string): void {
  deferred.delete(key)
}

function scheduleRetry(key: string, entry: DeferredEntry, retry: () => Promise<void>): void {
  setTimeout(() => {
    void Promise.resolve()
      .then(retry)
      .then(
        () => settle(key),
        (error: unknown) => {
          if (error instanceof OwnerWriteDeferredMarker) {
            // Still deferred: the re-entrant park call just refreshed this
            // entry and scheduled the next re-entry. The entry STAYS — the
            // attempt counter must never reset mid-chain.
            return
          }

          // Expired (the park call already deleted the entry) or a genuine
          // failure (the store ran its failure path). Either way: settled.
          settle(key)
        }
      )
      .catch(() => {})
  }, OWNER_DEFERRED_RETRY_MS)
}

/**
 * Park a refused sync write for retry, or report the deferral expired.
 *
 * Returns 'parked' while the write still has retry attempts left (scheduling
 * the next re-entry into `retry`), 'expired' once the bound is exhausted —
 * the caller then runs its ordinary failure semantics. Re-entering calls
 * refresh the retry thunk on the SAME entry, so a deferral can never be
 * extended or reset by re-creating it.
 */
export function parkUntilOwnerAvailable(key: string, retry: () => Promise<void>): 'expired' | 'parked' {
  let entry = deferred.get(key)

  if (entry && entry.attempts >= OWNER_DEFERRED_MAX_ATTEMPTS) {
    deferred.delete(key)

    return 'expired'
  }

  if (!entry) {
    entry = { attempts: 0 }
    deferred.set(key, entry)
  }

  entry.attempts += 1
  scheduleRetry(key, entry, retry)

  return 'parked'
}

/** Test access to the live deferral bookkeeping. */
export const _deferredEntries = deferred
