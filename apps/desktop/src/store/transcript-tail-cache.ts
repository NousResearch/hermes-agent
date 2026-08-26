import type { ChatMessage } from '@/lib/chat-messages'
import { withUniqueToolCallIdsWithinMessage } from '@/lib/chat-messages'

// ── Durable transcript-tail cache (#89206 "feels instant" layer) ────────────
// The in-memory warm cache (sessionStateByRuntimeIdRef) makes same-window
// re-opens instant, but dies with the window — so every app launch, backend
// reap, or profile respawn pays a full network round-trip before ANYTHING
// paints, and on a cold multi-profile boot that round-trip sits behind a
// backend spawn. This cache persists a bounded tail of each stored session's
// transcript in localStorage: a wake paints the cached tail at ~0ms, the
// paint-first hydration completes immediately, and the REST prefetch /
// runtime resume reconcile the authoritative transcript when they land
// (the existing reconcilers already treat painted content as "previous").
//
// Scope and bounds:
//   - TAIL ONLY (last TAIL_MESSAGES), size-capped per entry; oversized
//     messages evict the entry rather than truncating a message mid-parts.
//   - LRU across MAX_ENTRIES sessions; corrupt entries self-evict.
//   - Same trust domain as state.db on the same disk — no new exposure.
//   - Keyed by stored session id (durable identity), never runtime id,
//     SCOPED by the owning {connectionId, profile} (#94828). Stored ids are
//     only unique WITHIN one profile's state.db, and localStorage survives
//     profile switches in the same window: an unscoped key let profile A's
//     tail be painted against profile B's backend, which then retried a
//     session id that does not exist there ("session not found") on every
//     wake. The scope mirrors the in-memory twin's transcriptTailKey
//     (transcript-tail.ts). Entries persisted before scoping shipped (v1,
//     bare-id keys) carry no owner and are unreachable by construction —
//     they are swept once per window so a stale tail can never paint again.

const PREFIX = 'hermes.transcript-tail.v2:'
const LEGACY_ROOT = 'hermes.transcript-tail.v1'
const INDEX_KEY = 'hermes.transcript-tail.v2-index'
const TAIL_MESSAGES = 40
const MAX_ENTRY_BYTES = 256 * 1024
const MAX_ENTRIES = 50

/** Owning scope of an entry — same shape the REST layer threads through
 *  `getLatestSessionMessages(id, scope)` and the in-memory twin stores in
 *  each TranscriptTailState. */
export type TranscriptTailScope =
  | null
  | string
  | {
      connectionId?: null | string
      profile?: null | string
    }

interface CacheEntry {
  messages: ChatMessage[]
  savedAt: number
}

let legacyPurged = false

/** One-time sweep of pre-scoping v1 entries (#94828): a bare-id key cannot
 *  be attributed to a profile, so it must never paint again. Best effort —
 *  worst case a stale entry lingers unread until quota eviction. */
function purgeLegacyV1(store: Storage): void {
  if (legacyPurged) {
    return
  }

  try {
    const doomed: string[] = []

    for (let index = 0; index < store.length; index += 1) {
      const key = store.key(index)

      if (key && key.startsWith(LEGACY_ROOT)) {
        doomed.push(key)
      }
    }

    for (const key of doomed) {
      store.removeItem(key)
    }

    // Only latch on a completed sweep: a mid-sweep throw retries on the next
    // storage() touch instead of leaving a partial purge latched for the
    // window's lifetime.
    legacyPurged = true
  } catch {
    // best effort
  }
}

function storage(): Storage | null {
  try {
    const store = window.localStorage

    purgeLegacyV1(store)

    return store
  } catch {
    return null
  }
}

function normalizedScope(scope?: TranscriptTailScope): { connectionId: string; profile: string } | null {
  if (typeof scope === 'string') {
    return { connectionId: '', profile: scope.trim() || 'default' }
  }

  if (!scope) {
    return null
  }

  return {
    connectionId: String(scope.connectionId ?? '').trim(),
    profile: String(scope.profile ?? '').trim() || 'default'
  }
}

/** Index identity of an entry: the bare stored id (legacy/unscoped callers)
 *  or the JSON [connectionId, profile, storedId] composite. */
function entrySuffix(storedSessionId: string, scope?: TranscriptTailScope): string {
  const scoped = normalizedScope(scope)

  return scoped ? JSON.stringify([scoped.connectionId, scoped.profile, storedSessionId]) : storedSessionId
}

function readIndex(store: Storage): string[] {
  try {
    const raw = store.getItem(INDEX_KEY)
    const parsed = raw ? JSON.parse(raw) : []

    return Array.isArray(parsed) ? parsed.filter(id => typeof id === 'string') : []
  } catch {
    return []
  }
}

function writeIndex(store: Storage, ids: string[]): void {
  try {
    store.setItem(INDEX_KEY, JSON.stringify(ids))
  } catch {
    // Quota — drop the index; entries become orphaned and are rewritten lazily.
  }
}

function touchIndex(store: Storage, suffix: string): void {
  const ids = readIndex(store).filter(id => id !== suffix)
  ids.push(suffix)

  while (ids.length > MAX_ENTRIES) {
    const evicted = ids.shift()

    if (evicted) {
      try {
        store.removeItem(PREFIX + evicted)
      } catch {
        // best effort
      }
    }
  }

  writeIndex(store, ids)
}

/** Persist the tail of a session's transcript. No-op on empty/oversized.
 *  Pass the session's owning scope ({connectionId, profile}, or just the
 *  profile name) so the entry can only ever be read against that backend. */
export function saveTranscriptTail(
  storedSessionId: string,
  messages: ChatMessage[],
  scope?: TranscriptTailScope
): void {
  const id = (storedSessionId ?? '').trim()
  const store = storage()

  if (!store || !id || !Array.isArray(messages) || messages.length === 0) {
    return
  }

  const entry: CacheEntry = { messages: messages.slice(-TAIL_MESSAGES), savedAt: Date.now() }

  let serialized: string

  try {
    serialized = JSON.stringify(entry)
  } catch {
    return // non-serializable parts (live handles) — skip, never throw
  }

  if (serialized.length > MAX_ENTRY_BYTES) {
    // Retry with a shorter tail before giving up; a session dominated by a
    // few huge tool results still caches its recent turns.
    const shorter: CacheEntry = { messages: messages.slice(-8), savedAt: entry.savedAt }

    try {
      serialized = JSON.stringify(shorter)
    } catch {
      return
    }

    if (serialized.length > MAX_ENTRY_BYTES) {
      return
    }
  }

  const suffix = entrySuffix(id, scope)

  try {
    store.setItem(PREFIX + suffix, serialized)
    touchIndex(store, suffix)
  } catch {
    // Quota exceeded — evict everything and retry once (small cache >> stale cache).
    try {
      clearTranscriptTails()
      store.setItem(PREFIX + suffix, serialized)
      touchIndex(store, suffix)
    } catch {
      // Storage genuinely unavailable; instant paint just won't happen.
    }
  }
}

/** Cached tail for a stored session, or null. Corrupt entries self-evict.
 *  Only entries saved under the SAME owning scope are returned (#94828). */
export function loadTranscriptTail(storedSessionId: string, scope?: TranscriptTailScope): ChatMessage[] | null {
  const id = (storedSessionId ?? '').trim()
  const store = storage()

  if (!store || !id) {
    return null
  }

  let raw: null | string = null

  try {
    raw = store.getItem(PREFIX + entrySuffix(id, scope))
  } catch {
    return null
  }

  if (!raw) {
    return null
  }

  try {
    const parsed = JSON.parse(raw) as CacheEntry

    if (!parsed || !Array.isArray(parsed.messages) || parsed.messages.length === 0) {
      throw new Error('empty')
    }

    // Repair, don't just trust: a tail persisted by an older build (or by a
    // producer bug) can carry two `tool-call` parts with one `toolCallId`
    // inside a single message. This path paints DIRECTLY into the view, and a
    // poisoned entry is re-read identically on every launch — the "one pane
    // permanently broken" shape of #87857. Renaming at read keeps already-
    // affected installs from crash-looping forever on upgraded builds.
    return parsed.messages.map(withUniqueToolCallIdsWithinMessage)
  } catch {
    try {
      store.removeItem(PREFIX + entrySuffix(id, scope))
    } catch {
      // best effort
    }

    return null
  }
}

/** Drop one session's cached tail (session deleted / cache poisoned). With a
 *  scope, only that scope's entry is dropped — other backends' tails for the
 *  same stored id stay intact. */
export function dropTranscriptTail(storedSessionId: string, scope?: TranscriptTailScope): void {
  const id = (storedSessionId ?? '').trim()
  const store = storage()

  if (!store || !id) {
    return
  }

  try {
    const suffix = entrySuffix(id, scope)

    store.removeItem(PREFIX + suffix)
    writeIndex(
      store,
      readIndex(store).filter(entry => entry !== suffix)
    )
  } catch {
    // best effort
  }
}

/** Drop EVERY scope's entry for a stored id. Delete-path only: the save-path
 *  scope (ownerRoute) and the delete-path scope (the removed row's
 *  connection_id) are derived from different sources, so a shape drift
 *  between them would orphan the entry until LRU eviction (#94914 review,
 *  defect 1). A DELETED stored id cannot be legitimately reused, so sweeping
 *  all scopes is safe there — but never on the failed-resume path, where a
 *  same-id twin in another profile must keep its own tail. */
export function dropTranscriptTailEverywhere(storedSessionId: string): void {
  const id = (storedSessionId ?? '').trim()
  const store = storage()

  if (!store || !id) {
    return
  }

  try {
    const namesId = (entry: string): boolean => {
      if (entry === id) {
        return true
      }

      if (!entry.startsWith('[')) {
        return false
      }

      try {
        const parsed = JSON.parse(entry)

        return Array.isArray(parsed) && parsed[2] === id
      } catch {
        return false
      }
    }

    const index = readIndex(store)

    for (const entry of index.filter(namesId)) {
      store.removeItem(PREFIX + entry)
    }

    writeIndex(
      store,
      index.filter(entry => !namesId(entry))
    )
  } catch {
    // best effort
  }
}

/** Wipe the whole cache (connection/mode re-home, quota recovery). */
export function clearTranscriptTails(): void {
  const store = storage()

  if (!store) {
    return
  }

  for (const id of readIndex(store)) {
    try {
      store.removeItem(PREFIX + id)
    } catch {
      // best effort
    }
  }

  try {
    store.removeItem(INDEX_KEY)
  } catch {
    // best effort
  }
}
