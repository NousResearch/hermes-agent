import type { UsageStats } from '@/types/hermes'

// ── Durable last-known context occupancy ─────────────────────────────
// Measured occupancy is process-local (agent.compressor), so after a restart
// a restored session reads 0% until its live agent rebinds and reports. This
// cache keeps the last nonzero read per stored session so the meter paints
// stale-but-labeled numbers instead of a false empty.
//
// Same trust domain and rules as the transcript-tail cache: same disk as
// state.db, keyed by stored session id scoped to the owning
// (connectionId, profile), versioned by the session row's counters + model so
// a transcript that advanced elsewhere invalidates the paint. Restored reads
// are ALWAYS estimates (`context_source: 'restored'`) and only paint while no
// live breakdown exists — live data replaces them on arrival.

const PREFIX = 'hermes.context-usage.v1:'
const INDEX_KEY = 'hermes.context-usage.v1-index'
const MAX_ENTRIES = 50

export type ContextUsageScope = null | string | { connectionId?: null | string; profile?: null | string }

export interface ContextUsageVersion {
  input_tokens: number
  message_count: number
  model: null | string
  output_tokens: number
}

export interface RestoredContextUsage {
  context_estimated: true
  context_max: number
  context_percent: number
  context_source: 'restored'
  context_used: number
  model?: string
}

interface CacheEntry {
  savedAt: number
  usage: { context_max: number; context_percent: number; context_used: number; model?: string }
  version: { input_tokens: number; message_count: number; model: string; output_tokens: number }
}

function storage(): Storage | null {
  try {
    return window.localStorage
  } catch {
    return null
  }
}

function normalizedScope(scope?: ContextUsageScope): { connectionId: string; profile: string } | null {
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

function entrySuffix(storedSessionId: string, scope?: ContextUsageScope): string {
  const scoped = normalizedScope(scope)

  return scoped ? JSON.stringify([scoped.connectionId, scoped.profile, storedSessionId]) : storedSessionId
}

function readIndex(store: Storage): string[] {
  try {
    const parsed = JSON.parse(store.getItem(INDEX_KEY) ?? '[]')

    return Array.isArray(parsed) ? parsed.filter(id => typeof id === 'string') : []
  } catch {
    return []
  }
}

function writeIndex(store: Storage, ids: string[]): void {
  try {
    store.setItem(INDEX_KEY, JSON.stringify(ids))
  } catch {
    // Quota — drop the index; entries become orphaned and rewrite lazily.
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

function dropSuffix(store: Storage, suffix: string): void {
  try {
    store.removeItem(PREFIX + suffix)
    writeIndex(
      store,
      readIndex(store).filter(entry => entry !== suffix)
    )
  } catch {
    // best effort
  }
}

const isCounter = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value) && value >= 0

function sanitizeEntry(parsed: unknown): CacheEntry | null {
  if (!parsed || typeof parsed !== 'object') {
    return null
  }

  const entry = parsed as Partial<CacheEntry>
  const usage = entry.usage
  const version = entry.version

  if (
    !usage ||
    !isCounter(usage.context_max) ||
    usage.context_max <= 0 ||
    !isCounter(usage.context_used) ||
    usage.context_used <= 0 ||
    usage.context_used > usage.context_max ||
    !isCounter(usage.context_percent) ||
    usage.context_percent < 0 ||
    usage.context_percent > 100 ||
    !version ||
    !isCounter(version.input_tokens) ||
    !isCounter(version.message_count) ||
    !isCounter(version.output_tokens) ||
    typeof version.model !== 'string' ||
    !isCounter(entry.savedAt)
  ) {
    return null
  }

  return {
    savedAt: entry.savedAt as number,
    usage: {
      context_max: usage.context_max,
      context_percent: usage.context_percent,
      context_used: usage.context_used,
      ...(typeof usage.model === 'string' ? { model: usage.model } : {})
    },
    version: {
      input_tokens: version.input_tokens,
      message_count: version.message_count,
      model: version.model,
      output_tokens: version.output_tokens
    }
  }
}

/** Persist a last-known read. No-op on empties — a zero snapshot needs no
 *  cache, the default readout already says 0. */
export function saveContextUsageSnapshot(
  storedSessionId: string,
  usage: Pick<UsageStats, 'context_max' | 'context_percent' | 'context_used'> & { model?: string },
  scope?: ContextUsageScope,
  version?: ContextUsageVersion | null
): void {
  const id = (storedSessionId ?? '').trim()
  const store = storage()

  if (!store || !id || !version) {
    return
  }

  const max = usage.context_max ?? 0
  const used = usage.context_used ?? 0
  const percent = usage.context_percent ?? 0

  if (!(max > 0) || !(used > 0) || used > max || !(percent >= 0) || !(percent <= 100)) {
    return
  }

  if (
    !isCounter(version.input_tokens) ||
    !isCounter(version.message_count) ||
    !isCounter(version.output_tokens) ||
    (version.model !== null && typeof version.model !== 'string')
  ) {
    return
  }

  const suffix = entrySuffix(id, scope)
  const entry: CacheEntry = {
    savedAt: Date.now(),
    usage: {
      context_max: max,
      context_percent: percent,
      context_used: used,
      ...(typeof usage.model === 'string' ? { model: usage.model } : {})
    },
    version: {
      input_tokens: version.input_tokens,
      message_count: version.message_count,
      model: version.model ?? '',
      output_tokens: version.output_tokens
    }
  }

  try {
    store.setItem(PREFIX + suffix, JSON.stringify(entry))
    touchIndex(store, suffix)
  } catch {
    // Storage unavailable — the wake just won't have a restored number.
  }
}

/** Last-known read for a stored session, or null. Only entries whose version
 *  still matches the session row's counters are returned, and a mismatch
 *  evicts the entry so a moved-on transcript can never paint again. */
export function loadContextUsageSnapshot(
  storedSessionId: string,
  scope?: ContextUsageScope,
  version?: ContextUsageVersion | null
): RestoredContextUsage | null {
  const id = (storedSessionId ?? '').trim()
  const store = storage()

  if (!store || !id || !version) {
    return null
  }

  const suffix = entrySuffix(id, scope)
  let raw: null | string = null

  try {
    raw = store.getItem(PREFIX + suffix)
  } catch {
    return null
  }

  if (!raw) {
    return null
  }

  const entry = (() => {
    try {
      return sanitizeEntry(JSON.parse(raw))
    } catch {
      return null
    }
  })()

  if (
    !entry ||
    entry.version.input_tokens !== version.input_tokens ||
    entry.version.message_count !== version.message_count ||
    entry.version.model !== (version.model ?? '') ||
    entry.version.output_tokens !== version.output_tokens
  ) {
    dropSuffix(store, suffix)

    return null
  }

  return {
    context_estimated: true,
    context_max: entry.usage.context_max,
    context_percent: entry.usage.context_percent,
    context_source: 'restored',
    context_used: entry.usage.context_used,
    ...(entry.usage.model ? { model: entry.usage.model } : {})
  }
}

/** Drop one session's snapshot (session deleted). With a scope, only that
 *  scope's entry goes — other backends' snapshots for the same id survive. */
export function dropContextUsageSnapshot(storedSessionId: string, scope?: ContextUsageScope): void {
  const id = (storedSessionId ?? '').trim()
  const store = storage()

  if (!store || !id) {
    return
  }

  dropSuffix(store, entrySuffix(id, scope))
}

/** Drop EVERY scope's entry for a stored id. Delete-path only: same rationale
 *  as the transcript tail — the save-path scope and the delete-path scope are
 *  derived from different sources, so a shape drift between them would orphan
 *  the entry until LRU eviction. A deleted stored id is never reused, so a
 *  sweep is safe here — but never on the failed-resume path, where a twin in
 *  another profile must keep its snapshot. */
export function dropContextUsageSnapshotEverywhere(storedSessionId: string): void {
  const id = (storedSessionId ?? '').trim()
  const store = storage()

  if (!store || !id) {
    return
  }

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

  try {
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

/** Wipe the whole cache (connection re-home, quota recovery). */
export function clearContextUsageSnapshots(): void {
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
