/**
 * federated-session-index.ts
 *
 * Persistent index of the conversations each registry connection last
 * reported, so the unified Sessions list can keep showing a connection that is
 * not reachable right now.
 *
 * The unified list is assembled at read time: every refresh re-reads each
 * already-connected backend (mergeRemoteProfileSessions in main.ts), and a
 * connection with no pooled backend contributes zero rows — indistinguishable
 * from "no conversations". This module is the small local cache that closes
 * that gap. It remembers the rows `fetchRegistrySessionRows`
 * (profile-session-routing.ts) already tags with `connection_id` + `profile`,
 * per `(connectionId, profile)` owner, so opening an indexed row routes exactly
 * like a live one (owner routing is unchanged; the rows carry the same tags).
 *
 * Additive: the index only EXTENDS the live merged list for connections that
 * have no readable backend; it never replaces or reorders a row a live read
 * produced. It is fed by the background enumerator
 * (registry-session-enumerator.ts), which only runs for registries holding more
 * than one connection, so a single-connection registry never writes or reads it.
 *
 * Storage: one JSON file beside connections.json (userData/session-index.json),
 * written atomically (tmp + rename). Deliberately not SQLite: the index is a
 * bounded cache (MAX_OWNERS owners × MAX_ROWS_PER_OWNER rows), a rewrite is
 * O(kB), and the main process already owns several atomic-JSON stores, so this
 * keeps the electron layer free of native dependencies. The file holds session
 * titles/previews, so it is dropped per-connection (`retainConnections`) when a
 * connection leaves the registry, and wholesale by `clear()`.
 *
 * Kept electron-free (no `import 'electron'`) so it unit-tests under the
 * `electron` vitest project's node environment — same pattern as
 * connection-registry.ts.
 */

import * as fs from 'node:fs'
import * as path from 'node:path'

/** One indexed conversation. Only the fields the sidebar needs to paint a row
 * and route an open; the transcript itself is always fetched live from the
 * owning host. */
export interface IndexedSessionRow {
  id: string
  profile: string
  connection_id: string
  title?: null | string
  /** Unix seconds — the same key the read-time merge sorts by. */
  last_active?: number
  started_at?: number
  preview?: null | string
  [key: string]: unknown
}

/** File schema. `version` gates shape changes; readers reject unknown versions
 * rather than guessing (the index is a rebuildable cache — a rejected file is
 * simply treated as absent and re-accumulated). */
export interface FederatedSessionIndex {
  version: 1
  /** `connectionId::profile` → newest-first rows. */
  owners: Record<string, IndexedSessionRow[]>
}

export const FEDERATED_SESSION_INDEX_VERSION = 1 as const
export const SESSION_INDEX_FILE_NAME = 'session-index.json'

/** Bound the cache: an owner's rows are replaced wholesale by a fresh read
 * (mergeSessionIndexRows keeps only MAX_ROWS_PER_OWNER newest), and an owner
 * whose newest row is older than MAX_OWNER_IDLE_MS is dropped on the next
 * write. Keeps the file bounded no matter how many pages the enumerator reads. */
export const MAX_ROWS_PER_OWNER = 200
export const MAX_OWNERS = 50
export const MAX_OWNER_IDLE_MS = 14 * 24 * 60 * 60 * 1000 // 14 days

/** Row recency — the same key mergeRemoteProfileSessions sorts by. */
function rowRecency(row: IndexedSessionRow): number {
  return Math.max(Number(row.last_active) || 0, Number(row.started_at) || 0)
}

function ownerKey(connectionId: string, profile: string): string {
  return `${connectionId}::${profile}`
}

function rowsOfSessionList(data: unknown): IndexedSessionRow[] {
  if (!data || typeof data !== 'object' || !('sessions' in data)) {
    return []
  }

  return Array.isArray((data as { sessions?: unknown }).sessions)
    ? ((data as { sessions: unknown[] }).sessions as IndexedSessionRow[])
    : []
}

function rowOwnerOf(row: unknown): null | { connectionId: string; profile: string } {
  if (!row || typeof row !== 'object') {
    return null
  }

  const session = row as Record<string, unknown>
  const connectionId = typeof session.connection_id === 'string' ? session.connection_id.trim() : ''
  const profile = typeof session.profile === 'string' && session.profile.trim() ? session.profile.trim() : 'default'

  return connectionId ? { connectionId, profile } : null
}

/** Parse + shape-validate a session-index.json payload. Unknown version or
 * malformed owners → null (callers treat as absent; the cache rebuilds). */
export function parseSessionIndex(data: unknown): FederatedSessionIndex | null {
  if (
    !data ||
    typeof data !== 'object' ||
    (data as { version?: unknown }).version !== FEDERATED_SESSION_INDEX_VERSION
  ) {
    return null
  }

  const owners: Record<string, IndexedSessionRow[]> = {}
  const rawOwners = (data as { owners?: unknown }).owners

  if (rawOwners && typeof rawOwners === 'object') {
    for (const [key, value] of Object.entries(rawOwners as Record<string, unknown>)) {
      if (Array.isArray(value)) {
        owners[key] = value.filter(row => rowOwnerOf(row) !== null)
      }
    }
  }

  return { version: FEDERATED_SESSION_INDEX_VERSION, owners }
}

/** Newest-first per owner, at most MAX_ROWS_PER_OWNER. Sort BEFORE slicing:
 * an input page is not guaranteed recency-ordered, so slicing first would cap
 * by input position and drop the newest rows. */
function normalizeOwnerRows(rows: IndexedSessionRow[]): IndexedSessionRow[] {
  return [...rows].sort((a, b) => rowRecency(b) - rowRecency(a)).slice(0, MAX_ROWS_PER_OWNER)
}

/** Drop owners whose newest row is older than MAX_OWNER_IDLE_MS, and keep at
 * most MAX_OWNERS owners (newest activity wins). */
export function pruneSessionIndex(index: FederatedSessionIndex, now = Date.now()): FederatedSessionIndex {
  const nowSeconds = Math.floor(now / 1000)

  const owners = Object.entries(index.owners).filter(([, rows]) => {
    if (!rows.length) {
      return false
    }

    const newest = rows.reduce((max, row) => Math.max(max, rowRecency(row)), 0)

    return newest >= nowSeconds - MAX_OWNER_IDLE_MS / 1000
  })

  owners.sort((a, b) => rowRecency(b[1][0]) - rowRecency(a[1][0]))

  return { version: FEDERATED_SESSION_INDEX_VERSION, owners: Object.fromEntries(owners.slice(0, MAX_OWNERS)) }
}

/**
 * Remember the owner-tagged rows of one complete read of a connection. Rows
 * without a `connection_id` (the primary aggregate's own local rows) are
 * ignored: they are always re-readable from the primary backend. A successful
 * read REPLACES that connection's previous rows (a read is authoritative for
 * its connection), and an empty page is a successful read that clears them.
 * `now` stamps recency for pruning.
 */
export function mergeSessionIndexRows(
  index: FederatedSessionIndex,
  connectionId: string,
  sessions: IndexedSessionRow[] | { sessions?: unknown[] } | null | undefined,
  now = Date.now()
): FederatedSessionIndex {
  const rows = Array.isArray(sessions) ? sessions : rowsOfSessionList(sessions)
  const owners = { ...index.owners }

  // Group by the row's own profile tag (fetchRegistrySessionRows stamps it);
  // a shared-host read spans several profiles in one page.
  const byProfile = new Map<string, IndexedSessionRow[]>()

  for (const row of rows) {
    const owner = rowOwnerOf(row)

    if (!owner || owner.connectionId !== connectionId) {
      continue
    }

    const list = byProfile.get(owner.profile) ?? []

    list.push(row)
    byProfile.set(owner.profile, list)
  }

  for (const [profile, profileRows] of byProfile) {
    owners[ownerKey(connectionId, profile)] = normalizeOwnerRows(profileRows)
  }

  // The read is authoritative for the whole connection: a profile it did not
  // mention has no rows now (archived or deleted on the host), so clear any
  // rows still indexed for it.
  for (const key of Object.keys(owners)) {
    if (key.startsWith(`${connectionId}::`) && !byProfile.has(key.slice(connectionId.length + 2))) {
      delete owners[key]
    }
  }

  return pruneSessionIndex({ version: FEDERATED_SESSION_INDEX_VERSION, owners }, now)
}

/** Keep only the owners whose connection is still registered. */
export function retainIndexedConnections(
  index: FederatedSessionIndex,
  connectionIds: ReadonlySet<string>
): FederatedSessionIndex {
  const owners = Object.fromEntries(
    Object.entries(index.owners).filter(([key]) => connectionIds.has(key.slice(0, key.indexOf('::'))))
  )

  return Object.keys(owners).length === Object.keys(index.owners).length
    ? index
    : { version: FEDERATED_SESSION_INDEX_VERSION, owners }
}

/** All indexed rows across every owner, newest-first. */
export function indexedSessionRows(index: FederatedSessionIndex): IndexedSessionRow[] {
  const rows: IndexedSessionRow[] = []

  for (const ownerRows of Object.values(index.owners)) {
    rows.push(...ownerRows)
  }

  return rows.sort((a, b) => rowRecency(b) - rowRecency(a))
}

const csv = (value: null | string): string[] =>
  (value ?? '')
    .split(',')
    .map(part => part.trim())
    .filter(Boolean)

/**
 * Apply a `/api/profiles/sessions` request's filters to indexed rows, so a
 * backfilled row can only appear in a list the backend itself would have put
 * it in. The unified list is read as several slices (recents, cron, messaging)
 * with different filters; splicing unfiltered index rows into each would leak
 * cron and messaging rows into recents. The index only ever holds
 * non-archived rows (the enumerator reads `archived=exclude`), so
 * `archived=only` matches nothing.
 */
export function filterIndexedRows(rows: IndexedSessionRow[], searchParams: URLSearchParams): IndexedSessionRow[] {
  if (searchParams.get('archived') === 'only') {
    return []
  }

  const source = searchParams.get('source')?.trim() || ''
  const sources = csv(searchParams.get('sources'))
  const excluded = csv(searchParams.get('exclude_sources'))
  const minMessages = Math.max(0, Number(searchParams.get('min_messages')) || 0)

  return rows.filter(row => {
    const rowSource = typeof row.source === 'string' ? row.source : ''

    if (source && rowSource !== source) {
      return false
    }

    if (sources.length && !sources.includes(rowSource)) {
      return false
    }

    if (excluded.includes(rowSource)) {
      return false
    }

    return minMessages === 0 || (Number(row.message_count) || 0) >= minMessages
  })
}

/**
 * The registry connections whose rows the unified list cannot read live right
 * now: not the local connection and not the registry primary (their rows
 * always arrive through the primary aggregate), and no readable pooled
 * backend this refresh.
 */
export function coldRegistryConnectionIds(
  connections: ReadonlyArray<{ id: string; kind: string }>,
  primaryId: null | string | undefined,
  readableIds: ReadonlySet<string>
): Set<string> {
  return new Set(
    connections
      .filter(
        connection => connection.kind !== 'local' && connection.id !== primaryId && !readableIds.has(connection.id)
      )
      .map(connection => connection.id)
  )
}

/**
 * The indexed rows to splice into one unified-list read: only connections that
 * are cold right now, narrowed by the request's own filters.
 */
export function indexedRowsForColdConnections(
  index: FederatedSessionIndex,
  coldIds: ReadonlySet<string>,
  searchParams: URLSearchParams
): IndexedSessionRow[] {
  if (coldIds.size === 0) {
    return []
  }

  return filterIndexedRows(
    indexedSessionRows(index).filter(row => coldIds.has(row.connection_id)),
    searchParams
  )
}

/**
 * File-backed store. One instance lives in Electron main; `dir` is userData.
 * All operations are best-effort and never throw: the index is a cache, and a
 * read-only/full disk must not break the session list (same posture as the
 * registry's own write-failure handling).
 */
export class FederatedSessionIndexStore {
  private readonly filePath: string
  private cache: FederatedSessionIndex | null = null

  constructor(dir: string) {
    this.filePath = path.join(dir, SESSION_INDEX_FILE_NAME)
  }

  load(): FederatedSessionIndex {
    if (this.cache) {
      return this.cache
    }

    let parsed: unknown = null

    try {
      parsed = JSON.parse(fs.readFileSync(this.filePath, 'utf8'))
    } catch {
      // Absent or corrupt → empty (a rebuildable cache).
    }

    this.cache = parsed ? parseSessionIndex(parsed) : null

    return (this.cache ??= { version: FEDERATED_SESSION_INDEX_VERSION, owners: {} })
  }

  remember(connectionId: string, sessions: IndexedSessionRow[] | { sessions?: unknown[] } | null | undefined): void {
    try {
      this.persist(mergeSessionIndexRows(this.load(), connectionId, sessions))
    } catch {
      // Best-effort by contract.
    }
  }

  /** Forget every owner whose connection is no longer registered. */
  retainConnections(connectionIds: ReadonlySet<string>): void {
    try {
      const current = this.load()
      const next = retainIndexedConnections(current, connectionIds)

      if (next !== current) {
        this.persist(next)
      }
    } catch {
      // Best-effort by contract.
    }
  }

  rows(): IndexedSessionRow[] {
    return indexedSessionRows(this.load())
  }

  clear(): void {
    this.cache = { version: FEDERATED_SESSION_INDEX_VERSION, owners: {} }

    try {
      fs.rmSync(this.filePath, { force: true })
    } catch {
      // Best-effort.
    }
  }

  /** Atomic tmp+rename write. */
  private persist(index: FederatedSessionIndex): void {
    fs.mkdirSync(path.dirname(this.filePath), { recursive: true })
    const tmp = `${this.filePath}.tmp`
    fs.writeFileSync(tmp, JSON.stringify(index), 'utf8')
    fs.renameSync(tmp, this.filePath)
    this.cache = index
  }
}
