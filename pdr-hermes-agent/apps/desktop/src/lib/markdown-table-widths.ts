// Column widths for markdown tables in the transcript.
//
// A markdown table has no id — it is re-parsed from text on every render, and
// the same table reappears when the session is reloaded or switched back to. So
// the record is keyed by the table's own *shape*: its header cells and column
// count. That key survives re-render, streaming, session switches, and reloads
// without the transcript having to carry any resize state.
//
// Widths are stored as percentages of the table box, never pixels, so a
// restored table stays fluid and reads the same in a narrow pane as a wide one.
//
// The record is deliberately disposable: a small bounded namespace under one
// key, swept for expiry on first access. Losing it costs the user one drag.

import { readJson, writeJson } from './storage'

const STORAGE_KEY = 'hermes.desktop.mdTableColumns.v1'
const STORE_VERSION = 1
const MAX_AGE_MS = 7 * 24 * 60 * 60 * 1000
const MAX_ENTRIES = 64

interface WidthEntry {
  /** Column widths as percentages of the table box; sums to ~100. */
  w: number[]
  /** Last write, epoch ms — drives expiry and LRU eviction. */
  t: number
}

interface WidthStore {
  e: Record<string, WidthEntry>
  v: typeof STORE_VERSION
}

// In-memory mirror is the read path. It is seeded once from storage and stays
// authoritative for the life of the renderer, so a drag never pays a parse and
// a table restored mid-session sees the same value a reload would.
let cache: null | Record<string, WidthEntry> = null

function isEntry(value: unknown): value is WidthEntry {
  if (!value || typeof value !== 'object') {
    return false
  }

  const { t, w } = value as Partial<WidthEntry>

  return (
    typeof t === 'number' &&
    Array.isArray(w) &&
    w.length > 1 &&
    w.every(part => typeof part === 'number' && Number.isFinite(part) && part > 0)
  )
}

function load(): Record<string, WidthEntry> {
  if (cache) {
    return cache
  }

  const raw = readJson<Partial<WidthStore>>(STORAGE_KEY)
  const entries: Record<string, WidthEntry> = {}

  if (raw?.v === STORE_VERSION && raw.e && typeof raw.e === 'object') {
    const cutoff = Date.now() - MAX_AGE_MS

    for (const [key, entry] of Object.entries(raw.e)) {
      if (isEntry(entry) && entry.t > cutoff) {
        entries[key] = entry
      }
    }
  }

  cache = entries

  return entries
}

function flush(entries: Record<string, WidthEntry>) {
  const keys = Object.keys(entries)
  // Bounded namespace: drop the coldest tables past MAX_ENTRIES. `Math.max`
  // is not decoration — a negative count makes `slice` count from the END and
  // would evict nearly the whole store every time it fits comfortably.
  const excess = Math.max(0, keys.length - MAX_ENTRIES)

  for (const key of keys.sort((a, b) => entries[a].t - entries[b].t).slice(0, excess)) {
    delete entries[key]
  }

  writeJson(STORAGE_KEY, keys.length === excess ? null : { e: entries, v: STORE_VERSION })
}

/** Stable identity for a table, derived from its header row. Same hash shape as
 *  `profileColor` — cheap, and collisions only ever mean two identically-headed
 *  tables share a width record, which is the behaviour you'd want anyway. */
export function markdownTableKey(headers: string[]): string {
  let hash = 0
  const joined = `${headers.length}\u0001${headers.join('\u0001')}`

  for (let index = 0; index < joined.length; index += 1) {
    hash = (hash * 31 + joined.charCodeAt(index)) >>> 0
  }

  return hash.toString(36)
}

/** Stored percentages for a table, or null when it has never been resized (or
 *  the record no longer matches its column count). */
export function readTableWidths(key: string, columns: number): null | number[] {
  const entry = load()[key]

  return entry && entry.w.length === columns ? entry.w : null
}

export function writeTableWidths(key: string, widths: number[]) {
  const entries = load()
  entries[key] = { t: Date.now(), w: widths }
  flush(entries)
}

export function clearTableWidths(key: string) {
  const entries = load()

  if (entries[key]) {
    delete entries[key]
    flush(entries)
  }
}
