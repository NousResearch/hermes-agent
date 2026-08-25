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
//   - Keyed by stored session id (durable identity), never runtime id.

const PREFIX = 'hermes.transcript-tail.v1:'
const INDEX_KEY = 'hermes.transcript-tail.v1-index'
const TAIL_MESSAGES = 40
const MAX_ENTRY_BYTES = 256 * 1024
const MAX_ENTRIES = 50

interface CacheEntry {
  messages: ChatMessage[]
  savedAt: number
}

function storage(): Storage | null {
  try {
    return window.localStorage
  } catch {
    return null
  }
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

function touchIndex(store: Storage, storedSessionId: string): void {
  const ids = readIndex(store).filter(id => id !== storedSessionId)
  ids.push(storedSessionId)

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

/** Persist the tail of a session's transcript. No-op on empty/oversized. */
export function saveTranscriptTail(storedSessionId: string, messages: ChatMessage[]): void {
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

  try {
    store.setItem(PREFIX + id, serialized)
    touchIndex(store, id)
  } catch {
    // Quota exceeded — evict everything and retry once (small cache >> stale cache).
    try {
      clearTranscriptTails()
      store.setItem(PREFIX + id, serialized)
      touchIndex(store, id)
    } catch {
      // Storage genuinely unavailable; instant paint just won't happen.
    }
  }
}

/** Cached tail for a stored session, or null. Corrupt entries self-evict. */
export function loadTranscriptTail(storedSessionId: string): ChatMessage[] | null {
  const id = (storedSessionId ?? '').trim()
  const store = storage()

  if (!store || !id) {
    return null
  }

  let raw: null | string = null

  try {
    raw = store.getItem(PREFIX + id)
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
      store.removeItem(PREFIX + id)
    } catch {
      // best effort
    }

    return null
  }
}

/** Drop one session's cached tail (session deleted / cache poisoned). */
export function dropTranscriptTail(storedSessionId: string): void {
  const id = (storedSessionId ?? '').trim()
  const store = storage()

  if (!store || !id) {
    return
  }

  try {
    store.removeItem(PREFIX + id)
    writeIndex(
      store,
      readIndex(store).filter(entry => entry !== id)
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
