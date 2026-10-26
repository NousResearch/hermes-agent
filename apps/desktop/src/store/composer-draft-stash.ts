/**
 * The shared draft stash every window of the app reads and writes.
 *
 * Each window keeps its own copy of localStorage, and another window's write
 * reaches it some milliseconds later. A window that wrote its whole draft
 * list from that copy dropped or reverted a draft the other window had just
 * saved, and the other window took the result as "sent elsewhere".
 *
 * So every draft carries the time it was last saved, and a cleared draft
 * leaves a timestamped marker for a while. Merging two copies keeps the newer
 * entry for each key (`newerEntry`). A window that sees storage holding an
 * older entry than the one it saved writes its own back, so a write made from
 * a stale copy cannot stick. The cap keeps the most recently saved drafts,
 * by those times, so every window drops the same ones.
 *
 * The authoritative copy is one key (`DRAFT_STASH_STORAGE_KEY`) so a read never
 * sees half of another window's write. The older `{ key: text }` list is still
 * written after it, for builds and readers that only know that list; two
 * windows writing in the same instant can leave that mirror one write behind.
 */

export const DRAFT_STASH_STORAGE_KEY = 'hermes:composer-drafts:v4'

/** `text: null` marks a draft cleared at `at` (sent, emptied). */
export interface StashEntry {
  at: number
  text: string | null
}

export type StashEntries = Map<string, StashEntry>

interface StoredStash {
  cleared?: Record<string, number>
  drafts?: Record<string, { at?: number; text?: string }>
}

// A marker only has to outlive another window's stale copy (milliseconds);
// the cap bounds a long session's sends.
const CLEARED_MARKER_TTL_MS = 10 * 60_000
const MAX_CLEARED_MARKERS = 200

/** a wins over b: saved later, or (same millisecond) by a fixed order so every window agrees. */
export function newerEntry(a: StashEntry | undefined, b: StashEntry | undefined): boolean {
  if (!a) {
    return false
  }

  if (!b || a.at !== b.at) {
    return !b || a.at > b.at
  }

  return (a.text ?? '') > (b.text ?? '')
}

export function hasDraftStash(): boolean {
  try {
    return window.localStorage.getItem(DRAFT_STASH_STORAGE_KEY) !== null
  } catch {
    return false
  }
}

export function readDraftStash(): StashEntries {
  const entries: StashEntries = new Map()

  try {
    const stored = JSON.parse(window.localStorage.getItem(DRAFT_STASH_STORAGE_KEY) ?? '{}') as StoredStash

    for (const [key, at] of Object.entries(stored.cleared ?? {})) {
      entries.set(key, { at: Number(at) || 0, text: null })
    }

    for (const [key, draft] of Object.entries(stored.drafts ?? {})) {
      if (typeof draft?.text === 'string' && draft.text) {
        entries.set(key, { at: Number(draft.at) || 0, text: draft.text })
      }
    }
  } catch {
    // Unreadable reads as empty: every window then writes back what it saved.
  }

  return entries
}

/** The pre-v4 list: texts with no times, so every one counts as the oldest. */
export function readLegacyDrafts(legacyKey: string): StashEntries {
  try {
    const raw = window.localStorage.getItem(legacyKey)
    const texts = raw ? (JSON.parse(raw) as Record<string, string>) : {}

    return new Map(
      Object.entries(texts)
        .filter(([, text]) => typeof text === 'string' && text)
        .map(([key, text]) => [key, { at: 0, text }])
    )
  } catch {
    return new Map()
  }
}

/** Keep the `maxDrafts` most recently saved drafts and only recent cleared markers. */
export function compactDraftStash(entries: StashEntries, maxDrafts: number, now: number): StashEntries {
  const byRecency = [...entries].sort(([ka, a], [kb, b]) => b.at - a.at || (ka < kb ? -1 : ka > kb ? 1 : 0))
  const kept: StashEntries = new Map()
  let drafts = 0
  let markers = 0

  for (const [key, entry] of byRecency) {
    if (entry.text !== null) {
      if (drafts++ < maxDrafts) {
        kept.set(key, entry)
      }
    } else if (now - entry.at < CLEARED_MARKER_TTL_MS && markers++ < MAX_CLEARED_MARKERS) {
      kept.set(key, entry)
    }
  }

  return kept
}

/** Write the stash (and the legacy list after it); `false` when storage refused. */
export function writeDraftStash(entries: StashEntries, legacyKey: string): boolean {
  const drafts: NonNullable<StoredStash['drafts']> = {}
  const cleared: Record<string, number> = {}
  const texts: Record<string, string> = {}

  // Oldest first, so the legacy list keeps its most-recent-last order.
  for (const [key, entry] of [...entries].sort(([, a], [, b]) => a.at - b.at)) {
    if (entry.text === null) {
      cleared[key] = entry.at
    } else {
      drafts[key] = { at: entry.at, text: entry.text }
      texts[key] = entry.text
    }
  }

  try {
    const raw = JSON.stringify({ cleared, drafts })

    if (window.localStorage.getItem(DRAFT_STASH_STORAGE_KEY) !== raw) {
      window.localStorage.setItem(DRAFT_STASH_STORAGE_KEY, raw)
    }

    if (Object.keys(texts).length === 0) {
      window.localStorage.removeItem(legacyKey)
    } else {
      window.localStorage.setItem(legacyKey, JSON.stringify(texts))
    }

    return true
  } catch {
    return false
  }
}
