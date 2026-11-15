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
 * Times come from one clock across every key (`nextStashStamp`): the newest
 * stamp in the stash plus one when the wall clock reads earlier. After the
 * system clock jumps back, a draft saved then still sorts newest, so the cap
 * never evicts it on its own save.
 *
 * The authoritative copy is one key (`DRAFT_STASH_STORAGE_KEY`) so a read never
 * sees half of another window's write. The older `{ key: text }` list is still
 * written, before it, for older builds that only know that list. That list is
 * exactly the texts of the stash's drafts, so the stash is also the copy of the
 * list as this build last wrote it. An older build (after a downgrade) edits,
 * sends and adds drafts only in the list: at boot, a list that differs from
 * that copy was written there, and `adoptLegacyEdits` takes its changes (a key
 * gone from the list was sent or cleared there). Two windows writing in the
 * same instant can leave the list one write behind; the next stash write any
 * window hears about rewrites it (`healLegacyMirror`). Until then the list is
 * another window's mirror, not an older build's edit: every list this build
 * writes leaves a receipt keyed by its hash just before it, so a list with a
 * receipt is never adopted.
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
  /** `false`: the legacy list could not be written with this stash, so it is not a copy of it. */
  mirrored?: boolean
}

// A marker only has to outlive another window's stale copy (milliseconds);
// the cap bounds a long session's sends.
const CLEARED_MARKER_TTL_MS = 10 * 60_000
const MAX_CLEARED_MARKERS = 200
const MIRROR_RECEIPT_PREFIX = 'hermes:composer-drafts:v3-mirror:'
const MAX_MIRROR_RECEIPTS = 32

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

/** Later than every entry in `entries` and no earlier than the wall clock. */
export function nextStashStamp(entries: StashEntries, now: number): number {
  let highest = 0

  for (const entry of entries.values()) {
    highest = Math.max(highest, entry.at)
  }

  return Math.max(now, highest + 1)
}

function readStoredStash(): StoredStash {
  try {
    return JSON.parse(window.localStorage.getItem(DRAFT_STASH_STORAGE_KEY) ?? '{}') as StoredStash
  } catch {
    return {}
  }
}

export function readDraftStash(): StashEntries {
  const entries: StashEntries = new Map()

  try {
    const stored = readStoredStash()

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

function hashText(text: string): string {
  let a = 0x811c9dc5
  let b = 0x9e3779b9

  for (let index = 0; index < text.length; index++) {
    const code = text.charCodeAt(index)
    a = Math.imul(a ^ code, 0x01000193)
    b = Math.imul(b ^ code, 0x5bd1e995)
  }

  return `${(a >>> 0).toString(36)}${(b >>> 0).toString(36)}`
}

const mirrorReceiptKey = (raw: string | null) => MIRROR_RECEIPT_PREFIX + hashText(raw ?? '')

function readLegacyRaw(legacyKey: string): string | null | undefined {
  try {
    return window.localStorage.getItem(legacyKey)
  } catch {
    return undefined
  }
}

/** Record that this build is about to write `raw` as the list; drop receipts past the marker lifetime or the cap. */
function writeMirrorReceipt(raw: string | null, now: number): void {
  const storage = window.localStorage

  const receipts = Array.from({ length: storage.length }, (_, index) => storage.key(index))
    .filter((key): key is string => key?.startsWith(MIRROR_RECEIPT_PREFIX) ?? false)
    .map(key => [key, Number(storage.getItem(key))] as const)
    .sort(([, a], [, b]) => b - a)

  receipts.forEach(([key, at], index) => {
    if (index >= MAX_MIRROR_RECEIPTS - 1 || !(now - at < CLEARED_MARKER_TTL_MS)) {
      storage.removeItem(key)
    }
  })
  storage.setItem(mirrorReceiptKey(raw), String(now))
}

/** The list as it stands is one this build wrote recently (an unfinished mirror), not an older build's. */
function isOwnMirror(raw: string | null, now: number): boolean {
  try {
    return now - Number(window.localStorage.getItem(mirrorReceiptKey(raw)) ?? -Infinity) < CLEARED_MARKER_TTL_MS
  } catch {
    return false
  }
}

/** The pre-v4 list (absent reads as empty); `null` when it is unreadable. */
function parseLegacyDrafts(legacyKey: string): StashEntries | null {
  try {
    const raw = window.localStorage.getItem(legacyKey)
    const texts: unknown = raw ? JSON.parse(raw) : {}

    if (!texts || typeof texts !== 'object' || Array.isArray(texts)) {
      return null
    }

    return new Map(
      Object.entries(texts)
        .filter((entry): entry is [string, string] => typeof entry[1] === 'string' && entry[1] !== '')
        .map(([key, text]) => [key, { at: 0, text }])
    )
  } catch {
    return null
  }
}

/** The pre-v4 list: texts with no times, so every one counts as the oldest. */
export function readLegacyDrafts(legacyKey: string): StashEntries {
  return parseLegacyDrafts(legacyKey) ?? new Map()
}

/**
 * Keys whose text in the legacy list is not the stash's (a cleared marker reads
 * as absent); `null` when the list is unreadable, which proves nothing.
 */
function legacyChanges(entries: StashEntries, legacyKey: string): Map<string, string | null> | null {
  const legacy = parseLegacyDrafts(legacyKey)

  if (!legacy) {
    return null
  }

  const changes = new Map<string, string | null>()

  for (const key of new Set([...entries.keys(), ...legacy.keys()])) {
    const theirs = legacy.get(key)?.text ?? null

    if (theirs !== (entries.get(key)?.text ?? null)) {
      changes.set(key, theirs)
    }
  }

  return changes
}

/**
 * Take what an older build changed in the legacy list since this build last
 * wrote it: edited and added drafts, and keys it dropped (sent or cleared
 * there) as cleared markers, all stamped newest. `null` when nothing changed.
 */
export function adoptLegacyEdits(entries: StashEntries, legacyKey: string, now: number): StashEntries | null {
  const raw = readLegacyRaw(legacyKey)

  if (readStoredStash().mirrored === false || raw === undefined || isOwnMirror(raw, now)) {
    return null
  }

  const changes = legacyChanges(entries, legacyKey)

  if (!changes?.size) {
    return null
  }

  const at = nextStashStamp(entries, now)
  const adopted: StashEntries = new Map(entries)

  for (const [key, text] of changes) {
    adopted.set(key, { at, text })
  }

  return adopted
}

/** Rewrite the legacy list from the stored stash when another window's write left it behind. */
export function healLegacyMirror(legacyKey: string): void {
  const entries = readDraftStash()

  if (legacyChanges(entries, legacyKey)?.size !== 0) {
    writeLegacyList(entries, legacyKey)
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

/** The legacy `{ key: text }` list, oldest first (its most-recent-last order); `false` when storage refused. */
function writeLegacyList(entries: StashEntries, legacyKey: string): boolean {
  const texts: Record<string, string> = {}

  for (const [key, entry] of [...entries].sort(([, a], [, b]) => a.at - b.at)) {
    if (entry.text !== null) {
      texts[key] = entry.text
    }
  }

  const raw = Object.keys(texts).length === 0 ? null : JSON.stringify(texts)

  try {
    writeMirrorReceipt(raw, Date.now())

    if (raw === null) {
      window.localStorage.removeItem(legacyKey)
    } else {
      window.localStorage.setItem(legacyKey, raw)
    }

    return true
  } catch {
    return false
  }
}

/**
 * Write the legacy list, then the stash; `false` when storage refused the stash.
 * The list goes first so a stash write that fails leaves the list newer, and
 * the next boot adopts it (its receipt is dropped so it reads as unmirrored); a
 * list write that fails marks the stash `mirrored: false`, so the next boot
 * does not take the stale list for an older build's.
 */
export function writeDraftStash(entries: StashEntries, legacyKey: string): boolean {
  const drafts: NonNullable<StoredStash['drafts']> = {}
  const cleared: Record<string, number> = {}

  for (const [key, entry] of [...entries].sort(([, a], [, b]) => a.at - b.at)) {
    if (entry.text === null) {
      cleared[key] = entry.at
    } else {
      drafts[key] = { at: entry.at, text: entry.text }
    }
  }

  const mirrored = writeLegacyList(entries, legacyKey)

  try {
    const raw = JSON.stringify(mirrored ? { cleared, drafts } : { cleared, drafts, mirrored })

    if (window.localStorage.getItem(DRAFT_STASH_STORAGE_KEY) !== raw) {
      window.localStorage.setItem(DRAFT_STASH_STORAGE_KEY, raw)
    }

    return true
  } catch {
    try {
      window.localStorage.removeItem(mirrorReceiptKey(window.localStorage.getItem(legacyKey)))
    } catch {
      // Storage refused everything: the next boot reads what is left.
    }

    return false
  }
}
