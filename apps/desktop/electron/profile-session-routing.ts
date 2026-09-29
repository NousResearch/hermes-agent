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

function profileSessionId(row: unknown): string | null {
  const id = sessionId(row)

  if (!id) {
    return null
  }

  const profile =
    row && typeof row === 'object' && 'profile' in row && typeof row.profile === 'string' ? row.profile : ''

  return `${profile}\0${id}`
}

export function mergeProfileSessionWindow(rows: unknown[], offset: number, limit: number): unknown[] {
  const window = rows.slice(offset, offset + limit)
  const seenRows = new Set(window)
  const seenIds = new Set(window.map(profileSessionId).filter((id): id is string => id !== null))

  for (const row of rows.slice(offset + limit)) {
    if (!isPinned(row)) {
      continue
    }

    const id = profileSessionId(row)

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

  return {
    cron: slice('cron_limit', '50', { profile, source: 'cron' }),
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

      const total = nonNegativeNumber(
        page && typeof page === 'object' ? (page as { total?: unknown }).total : null
      )

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
