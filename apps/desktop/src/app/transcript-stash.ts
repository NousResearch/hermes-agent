import type { ChatMessage } from '@/lib/chat-messages'

// In-memory transcript LRU — the "instant switch" layer (Ace 2026-07-15:
// switching sessions should feel like Discord/Slack, and the blank-loader
// flash between transcripts has to go).
//
// Layer order on a switch:
//   1. THIS cache (sync, same-frame paint — no blank, no loader)
//   2. disk render cache (async IPC+disk, ~ms — cold-boot warm start)
//   3. REST prefetch / gateway resume (network — the truth, wholesale-replaces)
//
// Rows stored here are the CONVERTED ChatMessage shape (same contract as the
// disk cache after the writer-shape fix) so the paint path never converts.
// This is a paint accelerator, never an authority: entries are snapshots of
// whatever the live store last held for that session, and every live payload
// (prefetch/resume/stream) overwrites the paint wholesale (invariant I1).

// Sized to hold the full boot preload set (MAX_PRELOAD = 12 pinned+visible
// sessions) PLUS a handful of manually visited sessions without evicting
// either group. Rows are references shared with the store/disk-cache path —
// per-entry cost is one array of pointers; tens of MB worst case in aggregate.
export const MAX_STASH_ENTRIES = 20

const entries = new Map<string, ChatMessage[]>()

/** Snapshot rows for a session (called when leaving it). Empty rows clear. */
export function stashTranscript(storedSessionId: string | null | undefined, rows: ChatMessage[]): void {
  if (!storedSessionId) {
    return
  }

  if (rows.length === 0) {
    entries.delete(storedSessionId)

    return
  }

  // Refresh LRU position: delete + set moves the key to the newest slot.
  entries.delete(storedSessionId)
  entries.set(storedSessionId, rows)

  while (entries.size > MAX_STASH_ENTRIES) {
    const oldest = entries.keys().next().value

    if (oldest === undefined) {
      break
    }

    entries.delete(oldest)
  }
}

/** Synchronous read for the switch paint. Returns null on miss. */
export function readStashedTranscript(storedSessionId: string | null | undefined): ChatMessage[] | null {
  if (!storedSessionId) {
    return null
  }

  const rows = entries.get(storedSessionId)

  if (!rows || rows.length === 0) {
    return null
  }

  // Refresh LRU position on read.
  entries.delete(storedSessionId)
  entries.set(storedSessionId, rows)

  return rows
}

/** Deleted/renamed sessions must not repaint from a stale snapshot. */
export function cullStashedTranscript(storedSessionId: string | null | undefined): void {
  if (storedSessionId) {
    entries.delete(storedSessionId)
  }
}

/** Test hook + profile-swap hygiene: drop everything. */
export function clearStashedTranscripts(): void {
  entries.clear()
}

/** Test introspection. */
export function stashedTranscriptCount(): number {
  return entries.size
}
