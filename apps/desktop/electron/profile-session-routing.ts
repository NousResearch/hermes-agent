interface SessionListResponse {
  sessions: unknown[]
  total: number
  [key: string]: unknown
}

/** HTTP status an error carries (the REST helpers stamp `err.statusCode`), NaN when none. */
function httpStatusOf(error: unknown): number {
  return Number(error && typeof error === 'object' ? (error as { statusCode?: unknown }).statusCode : NaN)
}

/** True when a remote rejected a profile-scoped read because it cannot serve
 * that scope: 400 (invalid profile param) or 404 (profile does not exist).
 * Auth (401/403), transport, and 5xx failures are real errors — retrying
 * those would just relabel the failure. */
export function isRemoteProfileScopeError(error: unknown): boolean {
  const status = httpStatusOf(error)

  return status === 400 || status === 404
}

export interface ProfileSessionsResponse extends SessionListResponse {
  profile_totals: Record<string, number>
}

type FetchJsonForProfile = (profile: string | null, path: string) => Promise<unknown>

const REMOTE_SESSION_PAGE_LIMIT = 100

/** Whether a read asks for the cross-profile Sessions list used by grouping. */
export function isAllProfilesSessionListRequest(method: string | undefined, path: string | undefined): boolean {
  if ((method || 'GET').toUpperCase() !== 'GET' || !path) {
    return false
  }

  let url: URL

  try {
    url = new URL(path, 'http://desktop.local')
  } catch {
    return false
  }

  if (url.pathname === '/api/profiles/sessions') {
    return (url.searchParams.get('profile') || 'all').trim() === 'all'
  }

  return (
    url.pathname === '/api/profiles/sessions/sidebar' &&
    (url.searchParams.get('recents_profile') || 'all').trim() === 'all'
  )
}

/** Paths whose session data must follow the legacy remote owner route. The fenced
 *  edit reads `/mutation-snapshot` before its PATCH; both must reach the backend that
 *  owns the row, or the revision fence compares against the wrong store (#138905). */
export function isRemoteSessionRequestPath(pathname: string): boolean {
  return /^\/api\/sessions\/[^/]+(?:\/(?:messages|messages\/around|mutation-snapshot|timeline))?$/.test(pathname)
}

function rowsOf(data: unknown): unknown[] {
  if (!data || typeof data !== 'object' || !('sessions' in data)) {
    return []
  }

  return Array.isArray(data.sessions) ? data.sessions : []
}

function tagRowsWithConnection(rows: unknown[], connectionId: string): void {
  for (const row of rows) {
    if (row && typeof row === 'object') {
      const session = row as Record<string, unknown>
      session.connection_id = connectionId
    }
  }
}

/** Preserve the registry source that served a session REST response.
 *
 * A registry-pinned request is dispatched directly to that remote host, so its
 * own session rows naturally omit Desktop's synthetic `connection_id`. Without
 * restoring that provenance, a `profile: "default"` row later resumes through
 * the legacy local primary instead of the active registry gateway. */
export function tagRegistrySessionResponse(path: string, data: unknown, connectionId: string): unknown {
  if (!data || typeof data !== 'object') {
    return data
  }

  const pathname = path.split('?', 1)[0].replace(/\/+$/, '')

  if (pathname === '/api/sessions' || pathname === '/api/profiles/sessions') {
    tagRowsWithConnection(rowsOf(data), connectionId)

    return data
  }

  if (pathname === '/api/profiles/sessions/sidebar') {
    const response = data as Record<string, unknown>

    for (const key of ['recents', 'cron', 'messaging']) {
      tagRowsWithConnection(rowsOf(response[key]), connectionId)
    }

    return data
  }

  if (/^\/api\/sessions\/[^/]+$/.test(pathname)) {
    const session = data as Record<string, unknown>
    session.connection_id = connectionId
  }

  return data
}

function sessionId(row: unknown): string | null {
  if (!row || typeof row !== 'object' || !('id' in row)) {
    return null
  }

  return typeof row.id === 'string' ? row.id : null
}

function nonNegativeNumber(value: unknown): number | null {
  return typeof value === 'number' && Number.isInteger(value) && value >= 0 ? value : null
}

function isPinned(row: unknown): boolean {
  return Boolean(row && typeof row === 'object' && 'pinned' in row && row.pinned)
}

export function mergeProfileSessionWindow(rows: unknown[], offset: number, limit: number): unknown[] {
  const spawnedKey = (session: Record<string, unknown>, id: unknown): string | null => {
    if (typeof id !== 'string') {
      return null
    }

    const connection = typeof session.connection_id === 'string' ? session.connection_id.trim() : ''
    const profile = typeof session.profile === 'string' ? session.profile : ''

    return `${!connection || connection === 'local' ? 'local' : connection}\0${profile}\0${id}`
  }

  const roots = rows.filter(
    row => !row || typeof row !== 'object' || !('spawned_by_session_id' in row) || !row.spawned_by_session_id
  )

  const window = roots.slice(offset, offset + limit)
  const seenRows = new Set(window)

  const seenIds = new Set(
    window
      .map(row => (row && typeof row === 'object' ? spawnedKey(row as Record<string, unknown>, sessionId(row)) : null))
      .filter((id): id is string => id !== null)
  )

  for (const row of roots.slice(offset + limit)) {
    if (!isPinned(row)) {
      continue
    }

    const id = row && typeof row === 'object' ? spawnedKey(row as Record<string, unknown>, sessionId(row)) : null

    if ((id && seenIds.has(id)) || (!id && seenRows.has(row))) {
      continue
    }

    if (id) {
      seenIds.add(id)
    } else {
      seenRows.add(row)
    }

    window.push(row)
  }

  const visible = new Set<string>()

  for (const row of window) {
    if (!row || typeof row !== 'object') {
      continue
    }

    const session = row as Record<string, unknown>
    const ids = Array.isArray(session._lineage_ids) ? session._lineage_ids : [session.id]

    for (const id of ids) {
      const key = spawnedKey(session, id)

      if (key) {
        visible.add(key)
      }
    }
  }

  const pending = rows.filter(
    row => row && typeof row === 'object' && 'spawned_by_session_id' in row && row.spawned_by_session_id
  )

  while (pending.length) {
    const index = pending.findIndex(row => {
      const session = row as Record<string, unknown>

      return visible.has(spawnedKey(session, session.spawned_by_session_id) ?? '')
    })

    if (index === -1) {
      break
    }

    const [row] = pending.splice(index, 1)
    const session = row as Record<string, unknown>
    const ids = Array.isArray(session._lineage_ids) ? session._lineage_ids : [session.id]
    window.push(row)

    for (const id of ids) {
      const key = spawnedKey(session, id)

      if (key) {
        visible.add(key)
      }
    }
  }

  return window
}

export interface SidebarSessionSliceParams {
  cron: URLSearchParams
  messaging: URLSearchParams
  recents: URLSearchParams
}

/** Build the three remote-profile sidebar reads from one workspace scope. */
export function buildSidebarSessionSliceParams(searchParams: URLSearchParams): SidebarSessionSliceParams {
  const profile = (searchParams.get('recents_profile') || 'all').trim() || 'all'

  const slice = (limitKey: string, defaultLimit: string, extra: Record<string, string>) =>
    new URLSearchParams({
      limit: searchParams.get(limitKey) || defaultLimit,
      offset: '0',
      min_messages: '1',
      archived: 'exclude',
      order: 'recent',
      ...extra
    })

  const recents = slice('recents_limit', '20', { profile })
  const recentsExclude = searchParams.get('recents_exclude')

  if (recentsExclude) {
    recents.set('exclude_sources', recentsExclude)
  }

  const messaging = slice('messaging_limit', '100', { profile })
  const messagingExclude = searchParams.get('messaging_exclude')

  if (messagingExclude) {
    messaging.set('exclude_sources', messagingExclude)
  }

  const cron = slice('cron_limit', '50', { profile, source: 'cron' })
  const includeSpawned = searchParams.get('include_spawned')

  if (includeSpawned) {
    for (const params of [recents, cron, messaging]) {
      params.set('include_spawned', includeSpawned)
    }
  }

  return {
    cron,
    messaging,
    recents
  }
}

interface SessionScanError {
  profile: string
  error: string
}

function errorsOf(data: unknown): SessionScanError[] | undefined {
  const errors = data && typeof data === 'object' ? (data as { errors?: unknown }).errors : undefined

  return Array.isArray(errors) && errors.length ? (errors as SessionScanError[]) : undefined
}

/** Fetch the primary backend's profile-aware session slice. A failed read is
 *  still an empty page, but it carries `errors` naming the requested scope
 *  (`all` for the unified list), like the backend's failed profile scan, so the
 *  renderer keeps the rows it could not re-read instead of clearing them. */
export async function fetchPrimaryProfileSessions(
  searchParams: URLSearchParams,
  fetchJsonForProfile: FetchJsonForProfile
): Promise<ProfileSessionsResponse> {
  try {
    return (await fetchJsonForProfile(null, `/api/profiles/sessions?${searchParams}`)) as ProfileSessionsResponse
  } catch (error) {
    const profile = (searchParams.get('profile') || '').trim() || 'all'

    return {
      sessions: [],
      total: 0,
      profile_totals: {},
      errors: [{ profile, error: error instanceof Error ? error.message : String(error) }]
    }
  }
}

/** Reassemble the batched sidebar response from its three per-slice reads,
 *  keeping each slice's `errors` so a failed scan is never read as an
 *  authoritative empty slice. */
export function assembleSidebarSessionSlices(recents: unknown, cron: unknown, messaging: unknown) {
  const slice = (data: unknown) => {
    const errors = errorsOf(data)

    return { sessions: rowsOf(data), ...(errors ? { errors } : {}) }
  }

  const recentsSlice = recents as Partial<ProfileSessionsResponse> | undefined

  return {
    recents: {
      ...slice(recents),
      total: Number(recentsSlice?.total) || 0,
      profile_totals: recentsSlice?.profile_totals || {}
    },
    cron: slice(cron),
    messaging: {
      ...slice(messaging),
      total: Number((messaging as Partial<SessionListResponse> | undefined)?.total) || rowsOf(messaging).length
    },
    errors: []
  }
}

/** One CONNECTED (already-pooled) registry gateway whose sessions belong in the
 *  unified list. `backends` carries the resolved descriptors: one per pooled
 *  (connection, profile) pair for ssh-scoped sources, a single shared host for
 *  remote/cloud. The caller resolves descriptors — this module only fetches,
 *  tags, and dedupes, so it stays unit-testable. */
export interface RegistrySessionSource {
  connectionId: string
  /** 'ssh' and forced-local backends each run as one profile; anything else is
   *  a shared host serving every profile via ?profile=. */
  kind: string
  backends: Array<{ descriptor: unknown; profileLabel: null | string }>
}

type PinnedRegistrySessionSource = Pick<RegistrySessionSource, 'connectionId'> &
  Partial<Pick<RegistrySessionSource, 'backends' | 'kind'>>

/**
 * A pinned registry request may use the aggregate route only when that
 * gateway is part of the already-pooled sources. Otherwise the aggregate
 * would look healthy while silently omitting the gateway the renderer asked
 * for, so the caller must keep the normal direct route instead.
 *
 * The legacy local/primary route is represented by the base aggregate rather
 * than a registry source and is therefore handled by the caller separately.
 */
export function hasPinnedRegistrySessionSource(
  connectionId: string | null | undefined,
  profile: string | null | undefined,
  sources: readonly PinnedRegistrySessionSource[],
  baseCoversLocal = true
): boolean {
  const required = String(connectionId ?? '').trim()
  const selectedProfile = String(profile ?? '').trim()

  if (!required || (required === 'local' && baseCoversLocal)) {
    return true
  }

  const source = sources.find(candidate => candidate.connectionId === required)

  if (!source) {
    return false
  }

  // Shared remote and cloud hosts serve every profile from one backend. An
  // absent profile is also allowed because the route can still be an explicit
  // all-profiles request without an ambient renderer profile.
  if (!selectedProfile || selectedProfile === 'all' || !source.kind || !source.backends) {
    return true
  }

  if (source.kind !== 'ssh' && source.kind !== 'local') {
    return true
  }

  return source.backends.some(({ profileLabel }) => (profileLabel || 'default') === selectedProfile)
}

/** include the local registry source when the primary aggregate is remote */
export function shouldIncludeLocalRegistrySessionSource(
  connectionId: string | null | undefined,
  baseCoversLocal = true
): boolean {
  return Boolean(String(connectionId ?? '').trim()) || !baseCoversLocal
}

type GetJsonForDescriptor = (descriptor: unknown, path: string) => Promise<unknown>

/** Read one registry source without sending a page larger than the backend cap. */
async function fetchSessionRowsInPages(
  basePath: string,
  searchParams: URLSearchParams,
  getPage: (path: string) => Promise<unknown>
): Promise<unknown[] | null> {
  const requestedLimit = Number(searchParams.get('limit'))
  const requestedOffset = Number(searchParams.get('offset') || '0')

  const needsPaging =
    Number.isInteger(requestedLimit) &&
    requestedLimit > REMOTE_SESSION_PAGE_LIMIT &&
    Number.isInteger(requestedOffset) &&
    requestedOffset >= 0

  try {
    if (!needsPaging) {
      return rowsOf(await getPage(`${basePath}?${searchParams}`))
    }

    const sessions: unknown[] = []
    const backfilled: unknown[] = []
    const seenIds = new Set<string>()
    const backfilledIds = new Set<string>()
    let pageOffset = requestedOffset
    let targetOffset = requestedOffset + requestedLimit

    while (pageOffset < targetOffset) {
      const pageParams = new URLSearchParams(searchParams)
      const pageLimit = Math.min(REMOTE_SESSION_PAGE_LIMIT, targetOffset - pageOffset)
      pageParams.set('limit', String(pageLimit))
      pageParams.set('offset', String(pageOffset))

      const page = await getPage(`${basePath}?${pageParams}`)
      const pageRows = rowsOf(page)

      const total = nonNegativeNumber(page && typeof page === 'object' ? (page as { total?: unknown }).total : null)

      const windowedCount =
        total !== null ? Math.min(pageLimit, Math.max(0, total - pageOffset)) : Math.min(pageLimit, pageRows.length)

      for (const row of pageRows.slice(0, windowedCount)) {
        const id = sessionId(row)

        if (id && seenIds.has(id)) {
          continue
        }

        if (id) {
          seenIds.add(id)
          backfilledIds.delete(id)
        }

        sessions.push(row)
      }

      for (const row of pageRows.slice(windowedCount)) {
        const id = sessionId(row)

        if ((id && seenIds.has(id)) || (id && backfilledIds.has(id))) {
          continue
        }

        if (id) {
          backfilledIds.add(id)
        }

        backfilled.push(row)
      }

      if (total !== null) {
        targetOffset = Math.min(targetOffset, total)
      }

      pageOffset += pageLimit
    }

    for (const row of backfilled) {
      const id = sessionId(row)

      if (id && seenIds.has(id)) {
        continue
      }

      if (id) {
        seenIds.add(id)
      }

      sessions.push(row)
    }

    return sessions
  } catch {
    return null
  }
}

/**
 * Every connected registry gateway's session rows for the unified Sessions
 * list (#88880), tagged with the owning `connection_id` + remote `profile` so
 * the renderer can route an open through the connection-scoped gateway.
 *
 * Hidden rows stay hidden: these reads NEVER pass `include_hidden`, so the
 * backend's default `hidden = 0` filter applies — Bot Mode canonical chats
 * (persisted hidden) are excluded from the global list exactly as local ones
 * are. Do not add include_hidden here; the Bots view has its own scoped
 * browser for those.
 *
 * A dead or erroring gateway contributes nothing rather than breaking the
 * sidebar.
 */
export async function fetchRegistrySessionRows(
  sources: RegistrySessionSource[],
  searchParams: URLSearchParams,
  getJson: GetJsonForDescriptor
): Promise<unknown[]> {
  const rows: unknown[] = []

  const tag = (sourceRows: unknown[], connectionId: string, profileLabel: null | string) => {
    for (const row of sourceRows) {
      if (!row || typeof row !== 'object') {
        continue
      }

      const session = row as Record<string, unknown>

      if (profileLabel !== null) {
        session.profile = profileLabel
      } else if (typeof session.profile !== 'string' || !session.profile) {
        session.profile = 'default'
      }

      session.is_default_profile = false
      session.connection_id = connectionId
      rows.push(session)
    }
  }

  await Promise.all(
    sources.map(async source => {
      if (source.kind === 'ssh' || source.kind === 'local') {
        // Each ssh-scoped or forced-local backend serves its own state.db
        // natively.
        await Promise.all(
          source.backends.map(async ({ descriptor, profileLabel }) => {
            const params = new URLSearchParams(searchParams)
            params.delete('profile')

            const sourceRows = await fetchSessionRowsInPages('/api/sessions', params, path => getJson(descriptor, path))

            if (sourceRows) {
              tag(sourceRows, source.connectionId, profileLabel || 'default')
            }
          })
        )

        return
      }

      // Shared remote/cloud host: one cross-profile read returns every
      // profile's rows, each tagged with its owning remote profile.
      const shared = source.backends[0]

      if (!shared) {
        return
      }

      const params = new URLSearchParams(searchParams)
      params.set('profile', 'all')

      let sourceRows = await fetchSessionRowsInPages('/api/profiles/sessions', params, path =>
        getJson(shared.descriptor, path)
      )

      if (!sourceRows) {
        // Older remote without the aggregator: its own default-profile list.
        const flat = new URLSearchParams(searchParams)
        flat.delete('profile')
        sourceRows = await fetchSessionRowsInPages('/api/sessions', flat, path => getJson(shared.descriptor, path))
      }

      if (sourceRows) {
        tag(sourceRows, source.connectionId, null)
      }
    })
  )

  return rows
}

/** Splice registry-gateway rows into an already-merged unified list: dedupe by
 *  session id (a v1 remote-override splice may already carry a row), keep the
 *  recency sort, and extend the per-profile totals so truncation flags stay
 *  honest. Mutates and returns `merged`/`profileTotals` the way the v1 splice
 *  does. */
export function spliceRegistrySessionRows(
  merged: unknown[],
  registryRows: unknown[],
  profileTotals: Record<string, number>
): { added: number } {
  const seen = new Set(merged.map(sessionId).filter((id): id is string => id !== null))
  let added = 0

  for (const row of registryRows) {
    const id = sessionId(row)

    if (id && seen.has(id)) {
      continue
    }

    if (id) {
      seen.add(id)
    }

    merged.push(row)
    added += 1

    const profile =
      row && typeof row === 'object' && 'profile' in row && typeof row.profile === 'string' && row.profile
        ? row.profile
        : 'default'

    profileTotals[profile] = (profileTotals[profile] || 0) + 1
  }

  return { added }
}

/**
 * The remote-profile query scope for a per-profile override read: the remote
 * alias when the override is a managed SSH connection with a configured
 * `remoteProfile` (the remote knows THAT name, not the Desktop label), else
 * the Desktop profile name itself. Empty only for an empty profile — every
 * concrete scope, `default` included, is named on the wire so a multi-profile
 * backend opens that profile's state.db instead of its launch profile's.
 */
export function remoteProfileQueryScope(profile: string, remoteProfileAlias?: null | string): string {
  const configured = String(remoteProfileAlias || '').trim()

  if (configured && configured !== 'default') {
    return configured
  }

  return String(profile || '').trim()
}

/** Options for {@link fetchRemoteProfileSessions}. */
export interface RemoteProfileSessionsOptions {
  /** Managed-SSH `remoteProfile` mapping, when the remote knows the profile
   * under a different name than the Desktop label. */
  remoteProfileAlias?: null | string
}

/**
 * #64999: stamp a remote session list's rows WITHOUT manufacturing labels.
 * The remote's own `profile` stamp is authoritative — overwriting it with the
 * Desktop connection-scope name relabeled rows from the backend's other
 * profiles (one remote URL shared by `wife` and `dad` scopes showed the launch
 * profile's sessions under both). The legacy label only backfills rows an
 * older remote returned unowned.
 *
 * Mutates and returns `rows` in place, mirroring the old splice behavior.
 */
export function tagRemoteSessionRows(rows: unknown[], scope: string): unknown[] {
  for (const row of rows) {
    if (!row || typeof row !== 'object') {
      continue
    }

    const session = row as Record<string, unknown>
    const owned = typeof session.profile === 'string' && session.profile.trim() !== ''

    if (!owned) {
      session.profile = scope
    }

    if (session.profile === scope) {
      session.is_default_profile = false
    }
  }

  return rows
}

/**
 * #64999: a per-session read/mutation against a per-profile override must be
 * scoped for a multi-profile remote (an unscoped /api/sessions/{id} opens the
 * backend's launch-profile state.db, so a resume 4007s even though the row
 * exists — under its real owner). Returns the path with the owner scope
 * applied; a legacy single-profile scope (`''`) keeps the path bare.
 */
export function pathWithRemoteOwnerScope(path: string, scope: string): string {
  const scoped = String(scope || '').trim()

  if (!scoped) {
    return path
  }

  const url = new URL(path, 'http://hermes.local')
  url.searchParams.set('profile', scoped)

  return `${url.pathname}${url.search}${url.hash}`
}

/** The legacy per-profile remote route for one session request: every non-profile query
 *  param survives (pagination on reads; a DELETE's request_id/expected_revision/
 *  expected_generation live only in the query) and the owner scope replaces ?profile=. */
export function remoteOwnerRoutePath(pathname: string, searchParams: URLSearchParams, ownerScope: string): string {
  const params = new URLSearchParams(searchParams)
  params.delete('profile')
  const query = params.toString()

  return pathWithRemoteOwnerScope(query ? `${pathname}?${query}` : pathname, ownerScope)
}

export async function fetchRemoteProfileSessions(
  profile: string,
  searchParams: URLSearchParams,
  fetchJsonForProfile: FetchJsonForProfile,
  options: RemoteProfileSessionsOptions = {}
): Promise<SessionListResponse> {
  const params = new URLSearchParams(searchParams)
  // #64999: a per-profile override can point at a MULTI-profile backend — one
  // `hermes serve` hosting several profiles — and an unscoped /api/sessions
  // reads whichever profile the backend process was launched under. Name the
  // scope so the rows provably belong to it; the remote's own stamps then
  // carry the authoritative identity (main.ts no longer relabels).
  const scope = remoteProfileQueryScope(profile, options.remoteProfileAlias)

  const fetchPage = async (pageParams: URLSearchParams): Promise<SessionListResponse> => {
    pageParams.delete('profile')

    if (scope) {
      pageParams.set('profile', scope)

      try {
        return (await fetchJsonForProfile(profile, `/api/sessions?${pageParams}`)) as SessionListResponse
      } catch (error) {
        // A remote that rejects the scope (400 unknown profile / 404 profile
        // does not exist) proves it serves a single launch profile natively —
        // the legacy shape this code was written for. Fall back to the
        // remote's own database exactly as before; auth/transport/5xx errors
        // are real failures and propagate.
        if (!isRemoteProfileScopeError(error)) {
          throw error
        }

        pageParams.delete('profile')
      }
    }

    return (await fetchJsonForProfile(profile, `/api/sessions?${pageParams}`)) as SessionListResponse
  }

  const requestedLimit = Number(params.get('limit'))
  const requestedOffset = Number(params.get('offset') || '0')

  const needsPaging =
    Number.isInteger(requestedLimit) &&
    requestedLimit > REMOTE_SESSION_PAGE_LIMIT &&
    Number.isInteger(requestedOffset) &&
    requestedOffset >= 0

  if (!needsPaging) {
    return fetchPage(params)
  }

  const sessions: unknown[] = []
  const backfilled: unknown[] = []
  const seenIds = new Set<string>()
  const backfilledIds = new Set<string>()
  let firstPage: SessionListResponse | null = null
  let pageOffset = requestedOffset
  let targetOffset = requestedOffset + requestedLimit

  while (pageOffset < targetOffset) {
    const pageParams = new URLSearchParams(params)
    const pageLimit = Math.min(REMOTE_SESSION_PAGE_LIMIT, targetOffset - pageOffset)
    pageParams.set('limit', String(pageLimit))
    pageParams.set('offset', String(pageOffset))

    const page = await fetchPage(pageParams)
    firstPage ??= page

    const total = nonNegativeNumber(page.total)
    const pageRows = rowsOf(page)

    const windowedCount =
      total !== null ? Math.min(pageLimit, Math.max(0, total - pageOffset)) : Math.min(pageLimit, pageRows.length)

    // /api/sessions appends pinned rows that fall outside the requested
    // window. Keep those aside until all ordinary pages have been joined so
    // pagination preserves the same order as one larger request.
    for (const row of pageRows.slice(0, windowedCount)) {
      const id = sessionId(row)

      if (id && seenIds.has(id)) {
        continue
      }

      if (id) {
        seenIds.add(id)
        backfilledIds.delete(id)
      }

      sessions.push(row)
    }

    for (const row of pageRows.slice(windowedCount)) {
      const id = sessionId(row)

      if ((id && seenIds.has(id)) || (id && backfilledIds.has(id))) {
        continue
      }

      if (id) {
        backfilledIds.add(id)
      }

      backfilled.push(row)
    }

    if (total !== null) {
      targetOffset = Math.min(targetOffset, total)
    }

    pageOffset += pageLimit
  }

  for (const row of backfilled) {
    const id = sessionId(row)

    if (!id || backfilledIds.has(id)) {
      sessions.push(row)
    }
  }

  const total = nonNegativeNumber(firstPage?.total)

  return {
    ...(firstPage || {}),
    sessions,
    total: total ?? sessions.length,
    limit: requestedLimit,
    offset: requestedOffset
  }
}

/** Per-profile budget for one remote's session read inside a sidebar aggregate.
 * A healthy remote answers in well under a second; a dead URL hangs until the
 * 180s backend readiness deadline. The aggregate must not inherit that deadline
 * (#75712) — each remote gets this budget, then contributes an error entry
 * instead of rows. */
export const REMOTE_PROFILE_SESSION_BUDGET_MS = 10_000

/** One remote profile's settled contribution to a sidebar aggregate: rows when
 *  it answered inside its budget, otherwise a named error. Never a rejection —
 *  one dead remote must not take the other profiles' rows down with it. */
export interface RemoteProfileSessionsOutcome {
  profile: string
  list: SessionListResponse | null
  error: string | null
}

/**
 * #75712: fetch every remote profile's session list, each under its own
 * bounded budget, and settle each outcome instead of letting one unavailable
 * remote block (or silently vanish from) the whole sidebar aggregate. The
 * underlying fetch keeps running past its budget — a remote that is merely
 * still booting can answer on a later refresh.
 */
export async function settleRemoteProfileSessions(
  remoteProfiles: readonly string[],
  fetchRemote: (profile: string) => Promise<unknown>,
  options: { budgetMs?: number } = {}
): Promise<RemoteProfileSessionsOutcome[]> {
  const budgetMs = options.budgetMs ?? REMOTE_PROFILE_SESSION_BUDGET_MS

  return Promise.all(
    remoteProfiles.map(async profile => {
      let timer: ReturnType<typeof setTimeout> | undefined

      try {
        const list = (await Promise.race([
          fetchRemote(profile),
          new Promise<never>((_, reject) => {
            timer = setTimeout(
              () => reject(new Error(`remote profile "${profile}" did not respond within ${budgetMs}ms`)),
              budgetMs
            )
          })
        ])) as SessionListResponse

        return { profile, list, error: null }
      } catch (error) {
        return {
          profile,
          list: null,
          error: error instanceof Error ? error.message : String(error)
        }
      } finally {
        if (timer) {
          clearTimeout(timer)
        }
      }
    })
  )
}

/** Fold each settled remote outcome (#75712) into the unified aggregate:
 *  answering remotes contribute rows and their real per-profile total; a
 *  dead or still-pending remote drops its stale local total and is NAMED in
 *  the returned error entries instead of silently vanishing. Returns the
 *  updated grand total plus the collected errors so they can ride the
 *  response payload. */
export function applyRemoteProfileSessionOutcomes(
  outcomes: readonly RemoteProfileSessionsOutcome[],
  merged: unknown[],
  profileTotals: Record<string, number>,
  total: number
): { total: number; errors: Array<{ profile: string; error: string }> } {
  const errors: Array<{ profile: string; error: string }> = []

  for (const { profile, list, error } of outcomes) {
    if (!list) {
      delete profileTotals[profile] // dead remote → drop its stale local total too
      errors.push({ profile, error: error || 'unavailable' })

      continue
    }

    const rows = rowsOf(list)
    merged.push(...rows)
    profileTotals[profile] = Number(list.total) || rows.length
    total += profileTotals[profile]
  }

  return { total, errors }
}

/** Compose the unified aggregate's response (#75712): failures ride the
 *  payload, not the void — base scan errors and each dead remote's named
 *  error entry are merged into `errors` so a failed remote profile is
 *  identifiable. */
export function composeUnifiedSessionResponse(
  base: unknown,
  merged: unknown[],
  offset: number,
  limit: number,
  total: number,
  profileTotals: Record<string, number>,
  extraErrors: Array<{ profile: string; error: string }>
) {
  const baseErrors = Array.isArray((base as { errors?: unknown })?.errors)
    ? (base as { errors: Array<{ profile: string; error: string }> }).errors
    : []

  const scanErrors = [...baseErrors, ...extraErrors]

  return {
    ...(base as Record<string, unknown>),
    sessions: mergeProfileSessionWindow(merged, offset, limit),
    total,
    profile_totals: profileTotals,
    ...(scanErrors.length ? { errors: scanErrors } : {})
  }
}

/**
 * #85834: which remote profile owns `sessionId`, when a /api/sessions/{id}
 * caller supplied no profile hint. Reads the same per-remote lists the list
 * endpoints splice into the sidebar (each fetch is per-profile, so a hit IS
 * the owner). Dead remotes contribute nothing; returns null when no remote
 * lists the id — the intercept then falls through to the local backend
 * exactly as before.
 */
export async function findRemoteOwnerProfileForSession(
  sessionId: string,
  remoteProfiles: readonly string[],
  listForProfile: (profile: string, searchParams: URLSearchParams) => Promise<SessionListResponse | null>
): Promise<null | string> {
  if (!sessionId || remoteProfiles.length === 0) {
    return null
  }

  const params = new URLSearchParams()
  params.set('limit', '200')
  params.set('offset', '0')

  const matches = await Promise.all(
    remoteProfiles.map(async profile => {
      const list = await listForProfile(profile, params).catch(() => null)
      const rows = Array.isArray(list?.sessions) ? (list.sessions as Array<Record<string, unknown>>) : []

      return rows.some(row => row?.id === sessionId || row?._lineage_root_id === sessionId) ? profile : null
    })
  )

  return matches.find(profile => profile !== null) ?? null
}

// The backend's `/api/profiles/sessions` page cap (FastAPI `le=500`).
const PROFILE_SESSION_PAGE_CAP = 500

const SIDEBAR_SLICES = ['recents', 'cron', 'messaging'] as const

type FetchProfilePage = (profile: string, path: string) => Promise<unknown>

type ProfileRead = { profile: string; data: any } | { profile: string; error: string }

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

function recencyOf(row: any): number {
  return Number(row?.last_active || row?.started_at || 0)
}

function sortedWindow(rows: unknown[], key: (row: any) => number, offset: number, limit: number): unknown[] {
  return mergeProfileSessionWindow([...rows].sort((a, b) => key(b) - key(a)), offset, limit)
}

function failedProfiles(errors: SessionScanError[]): Record<string, { failed: true; retry: true; error: string }> {
  return Object.fromEntries(
    errors.filter(err => err.profile).map(err => [err.profile, { failed: true, retry: true, error: err.error || '' }])
  )
}

async function readEachProfile(
  profiles: readonly string[],
  read: (profile: string) => Promise<unknown>
): Promise<ProfileRead[]> {
  return Promise.all(
    profiles.map(async profile => {
      try {
        return { profile, data: (await read(profile)) as any }
      } catch (error) {
        return { profile, error: errorMessage(error) }
      }
    })
  )
}

/** One profile's rows from offset 0 through `want`, in pages the backend accepts. The first
 *  page carries the profile's totals. */
async function readProfileSessionRows(url: URL, profile: string, want: number, fetchPage: FetchProfilePage) {
  let first: any
  const rows: unknown[] = []

  for (let offset = 0; ; ) {
    const params = new URLSearchParams(url.searchParams)
    const limit = Math.min(PROFILE_SESSION_PAGE_CAP, Math.max(0, want - offset))
    params.set('limit', String(limit))
    params.set('offset', String(offset))
    params.set('profile', profile)

    const page = (await fetchPage(profile, `${url.pathname}?${params}`)) as any
    const pageRows = rowsOf(page)
    first ??= page
    rows.push(...pageRows)
    offset += limit

    if (offset >= want || pageRows.length < limit) {
      return { ...first, sessions: rows }
    }
  }
}

async function allProfilesSessionList(url: URL, profiles: readonly string[], fetchPage: FetchProfilePage) {
  const limit = Math.max(0, Number(url.searchParams.get('limit') ?? 20) || 0)
  const offset = Math.max(0, Number(url.searchParams.get('offset')) || 0)
  const created = url.searchParams.get('order') === 'created'
  const reads = await readEachProfile(profiles, profile => readProfileSessionRows(url, profile, limit + offset, fetchPage))

  const rows: unknown[] = []
  const totals: Record<string, number> = {}
  const errors: SessionScanError[] = []
  const storage: Record<string, unknown> = {}

  for (const read of reads) {
    if ('error' in read) {
      errors.push({ profile: read.profile, error: read.error })

      continue
    }

    rows.push(...rowsOf(read.data))
    Object.assign(totals, read.data?.profile_totals)
    Object.assign(storage, read.data?.storage)
    errors.push(...(errorsOf(read.data) ?? []))
  }

  const key = created ? (row: any) => Number(row?.started_at || 0) : recencyOf

  return {
    sessions: sortedWindow(rows, key, offset, limit),
    total: Object.values(totals).reduce((sum, n) => sum + (Number(n) || 0), 0),
    profile_totals: totals,
    limit,
    offset,
    errors,
    storage
  }
}

async function allProfilesSidebar(url: URL, profiles: readonly string[], fetchPage: FetchProfilePage) {
  const cap = (key: string, fallback: number) =>
    Math.min(Math.max(Number(url.searchParams.get(key)) || fallback, 1), PROFILE_SESSION_PAGE_CAP)

  const caps = { recents: cap('recents_limit', 20), cron: cap('cron_limit', 50), messaging: cap('messaging_limit', 100) }

  const reads = await readEachProfile(profiles, profile => {
    const params = new URLSearchParams(url.searchParams)
    params.set('recents_profile', profile)

    // A registry-routed request also carries its route's `profile`; the gateway checks every
    // selector against the grant, so each read names only the profile it reads.
    if (params.has('profile')) {
      params.set('profile', profile)
    }

    return fetchPage(profile, `${url.pathname}?${params}`)
  })

  const rows: Record<(typeof SIDEBAR_SLICES)[number], unknown[]> = { recents: [], cron: [], messaging: [] }
  const truncated: Record<string, boolean> = {}
  const usage: Record<string, unknown> = {}
  const errors: SessionScanError[] = []
  const storage: Record<string, unknown> = {}
  const failed: SessionScanError[] = []
  let messagingTotal = 0
  let contributed = false

  for (const read of reads) {
    if ('error' in read) {
      errors.push({ profile: read.profile, error: read.error })
      failed.push({ profile: read.profile, error: read.error })

      continue
    }

    const data = read.data
    Object.assign(storage, data?.storage)
    errors.push(...(errorsOf(data) ?? []))

    // A failed slice has no `sessions` key: the profile's scan failed, it is not empty.
    if (data?.recents?.failed) {
      failed.push(...(errorsOf(data.recents) ?? [{ profile: read.profile, error: '' }]))

      continue
    }

    for (const err of Object.entries(data?.profiles_failed ?? {})) {
      failed.push({ profile: err[0], error: String((err[1] as any)?.error ?? '') })
    }

    for (const key of SIDEBAR_SLICES) {
      rows[key].push(...rowsOf(data?.[key]))
    }

    contributed ||= SIDEBAR_SLICES.some(key => rowsOf(data?.[key]).length > 0)
    Object.assign(truncated, data?.recents?.profiles_truncated)
    Object.assign(usage, data?.recents?.profiles_usage)
    messagingTotal += Number(data?.messaging?.total) || rowsOf(data?.messaging).length
  }

  const live = profiles.filter(profile => !(profile in storage))

  if (live.length && !contributed && live.every(profile => failed.some(err => err.profile === profile))) {
    const failedSlice = (extra = {}) => ({ failed: true, retry: true, errors: failed.map(err => ({ ...err })), ...extra })

    return {
      recents: failedSlice({ profiles_truncated: {}, profiles_usage: {} }),
      cron: failedSlice(),
      messaging: failedSlice({ total: 0 }),
      errors,
      storage
    }
  }

  const body: Record<string, any> = {
    recents: {
      sessions: sortedWindow(rows.recents, recencyOf, 0, caps.recents),
      profiles_truncated: truncated,
      profiles_usage: usage
    },
    cron: { sessions: sortedWindow(rows.cron, recencyOf, 0, caps.cron) },
    messaging: { sessions: sortedWindow(rows.messaging, recencyOf, 0, caps.messaging), total: messagingTotal },
    errors,
    storage
  }

  const profilesFailed = failedProfiles(failed)

  if (Object.keys(profilesFailed).length) {
    body.profiles_failed = profilesFailed

    for (const key of SIDEBAR_SLICES) {
      body[key].profiles_failed = profilesFailed
      body[key].errors = failed.map(err => ({ ...err }))
    }
  }

  return body
}

/** An all-profiles session list (`/api/profiles/sessions` or the batched sidebar) assembled
 *  from one read per profile.
 *
 *  A native gateway grant names ONE profile and the gateway refuses `all`, so a local native
 *  backend cannot answer the unified list in one request. Each profile is read with its own
 *  grant (`fetchPage` scopes the path to it) and merged the way the backend merges its own
 *  per-profile scans: recency order, pinned back-fill, per-profile totals, and a profile that
 *  cannot be read reported as a failed profile rather than as an empty one. */
export async function fetchAllProfilesSessionsPerProfile(
  path: string,
  profiles: readonly string[],
  fetchPage: FetchProfilePage
): Promise<unknown> {
  const url = new URL(path, 'http://desktop.local')

  return url.pathname.replace(/\/+$/, '') === '/api/profiles/sessions/sidebar'
    ? allProfilesSidebar(url, profiles, fetchPage)
    : allProfilesSessionList(url, profiles, fetchPage)
}

/** Cross-profile reads the gateway builds by scanning every profile: the all-profiles project
 *  tree, the PR scan and the cron job list. The renderer asks for the first two only for the
 *  all-profiles view, so a `profile` already on their path is the route's own scope, not a
 *  selector; the cron list is cross-profile only when it names `all` or no profile. The
 *  per-profile reads carry their own scope and are made with `profileScoped`. */
export function isAllProfilesAggregateRequest(method: string | undefined, path: string | undefined): boolean {
  if (!path) {
    return false
  }

  let url: URL

  try {
    url = new URL(path, 'http://desktop.local')
  } catch {
    return false
  }

  const pathname = url.pathname.replace(/\/+$/, '')
  const verb = (method || 'GET').toUpperCase()

  return (
    (verb === 'GET' && pathname === '/api/profiles/projects/tree') ||
    (verb === 'POST' && pathname === '/api/profiles/sessions/pull-requests') ||
    (verb === 'GET' && pathname === '/api/cron/jobs' && (url.searchParams.get('profile') || 'all').trim().toLowerCase() === 'all')
  )
}

function mergeTreeById(into: Map<string, any>, entries: any[], childKey: string): void {
  for (const entry of entries) {
    const existing = into.get(entry.id)

    if (!existing) {
      into.set(entry.id, entry)

      continue
    }

    if (childKey === 'sessions') {
      existing.sessions = [...(existing.sessions ?? []), ...(entry.sessions ?? [])]
    } else {
      const children = new Map<string, any>((existing[childKey] ?? []).map((child: any) => [child.id, child]))
      mergeTreeById(children, entry[childKey] ?? [], 'sessions')
      existing[childKey] = [...children.values()]
    }

    if ('sessionCount' in existing) {
      existing.sessionCount = (existing.sessionCount || 0) + (entry.sessionCount || 0)
    }
  }
}

// One folder's group from a second profile folded into the first: repos/groups by id, totals added,
// the newest activity and the newest previews across both.
function mergeProjectInto(existing: any, project: any, previewLimit: number) {
  const repos = new Map<string, any>((existing.repos ?? []).map((repo: any) => [repo.id, repo]))
  mergeTreeById(repos, project.repos ?? [], 'groups')
  existing.repos = [...repos.values()]

  for (const total of ['sessionCount', 'totalTokens', 'totalCostUsd']) {
    existing[total] = (existing[total] || 0) + (project[total] || 0)
  }

  existing.lastActive = Math.max(existing.lastActive || 0, project.lastActive || 0)
  existing.previewSessions = [...(existing.previewSessions ?? []), ...(project.previewSessions ?? [])]
    .sort((a, b) => recencyOf(b) - recencyOf(a))
    .slice(0, previewLimit)
}

// The gateway's own cross-profile merge (`_merge_profile_tree`), keyed by the folder key each
// per-profile read reports: one group per folder, a declared project wins over another
// profile's auto entry, totals add up, and the previews are the newest across profiles.
function mergeProfileProjectTrees(reads: ProfileRead[], previewLimit: number) {
  const merged = new Map<string, any>()
  const scopedSessionIds: unknown[] = []
  const errors: SessionScanError[] = []

  for (const read of reads) {
    if ('error' in read) {
      errors.push({ profile: read.profile, error: read.error })

      continue
    }

    scopedSessionIds.push(...(Array.isArray(read.data?.scoped_session_ids) ? read.data.scoped_session_ids : []))
    errors.push(...(errorsOf(read.data) ?? []))

    for (const incoming of Array.isArray(read.data?.projects) ? read.data.projects : []) {
      let project = incoming
      const key = String(project.mergeKey ?? project.path ?? project.id)
      let existing = merged.get(key)

      if (!existing) {
        merged.set(key, project)

        continue
      }

      if (existing.isAuto && !project.isAuto) {
        ;[existing, project] = [project, existing]
        merged.set(key, existing)
      }

      mergeProjectInto(existing, project, previewLimit)
    }
  }

  const projects = [...merged.values()].sort((a, b) => (b.lastActive || 0) - (a.lastActive || 0))

  return { projects, active_id: null, scoped_session_ids: scopedSessionIds, errors }
}

function mergeProfilePullRequests(reads: ProfileRead[]) {
  const pullRequests: Record<string, unknown> = {}
  let scanned: unknown[] = []

  // Profile order, like the gateway's scan: a later profile's match wins.
  for (const read of reads) {
    if (!('error' in read)) {
      Object.assign(pullRequests, read.data?.pull_requests)
      scanned = Array.isArray(read.data?.scanned) ? read.data.scanned : scanned
    }
  }

  // `scanned` lets the caller stop asking about an id. An unread profile scanned nothing, so
  // only the ids that did resolve are settled.
  return {
    pull_requests: pullRequests,
    scanned: reads.some(read => 'error' in read) ? Object.keys(pullRequests) : scanned
  }
}

// The gateway's own cross-profile list (`_list_cron_jobs_sync`): a job copied into a second
// profile is listed once, the default profile's copy winning, and a job without an id is never
// folded. A profile that cannot be read is left out, as the gateway logs and skips it.
function mergeProfileCronJobs(reads: ProfileRead[]) {
  const byId = new Map<string, any>()
  const unkeyed: unknown[] = []

  for (const read of reads) {
    if ('error' in read || !Array.isArray(read.data)) {
      continue
    }

    for (const job of read.data) {
      if (!job || typeof job !== 'object') {
        continue
      }

      const id = job.id || job.job_id

      if (!id || id === 'unknown') {
        unkeyed.push(job)

        continue
      }

      const existing = byId.get(id)

      if (!existing || (job.is_default_profile && !existing.is_default_profile)) {
        byId.set(id, job)
      }
    }
  }

  return [...byId.values(), ...unkeyed]
}

/** The all-profiles project tree, PR scan or cron list, assembled from one read per profile.
 *
 *  A native gateway grant names ONE profile, so each profile is read with its own grant
 *  (`fetchProfile` gets the path scoped to it). A profile that cannot be read is reported in
 *  `errors`; when none can be read the call fails, as the single request would have. */
export async function fetchAllProfilesAggregatePerProfile(
  path: string,
  profiles: readonly string[],
  fetchProfile: FetchProfilePage
): Promise<unknown> {
  const url = new URL(path, 'http://desktop.local')

  const reads = await readEachProfile(profiles, profile => {
    const params = new URLSearchParams(url.searchParams)
    params.set('profile', profile)

    return fetchProfile(profile, `${url.pathname}?${params}`)
  })

  if (!reads.some(read => !('error' in read))) {
    throw new Error(reads.map(read => ('error' in read ? `${read.profile}: ${read.error}` : '')).join('; ') || 'No profiles to read')
  }

  const pathname = url.pathname.replace(/\/+$/, '')

  if (pathname === '/api/profiles/projects/tree') {
    return mergeProfileProjectTrees(reads, Math.max(0, Number(url.searchParams.get('preview_limit') ?? 3) || 0))
  }

  if (pathname === '/api/cron/jobs') {
    return mergeProfileCronJobs(reads)
  }

  return mergeProfilePullRequests(reads)
}
