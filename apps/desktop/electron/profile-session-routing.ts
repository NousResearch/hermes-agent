interface SessionListResponse {
  sessions: unknown[]
  total: number
  [key: string]: unknown
}

export interface ProfileSessionsResponse extends SessionListResponse {
  profile_totals: Record<string, number>
}

type FetchJsonForProfile = (profile: string | null, path: string) => Promise<unknown>

const REMOTE_SESSION_PAGE_LIMIT = 100

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

export type ProfileSessionAggregateRoute = 'global-remote' | 'primary'

export interface ProfileSessionAggregateRouteOptions {
  globalRemote: boolean
  primaryProfileRemoteOverride: boolean
}

export interface ProfileSessionAggregateFetchOptions extends ProfileSessionAggregateRouteOptions {
  fetchJsonForGlobalRemote: FetchJsonForProfile
}

// The primary backend normally owns aggregate profile reads. In mixed remote
// mode, however, the active profile's explicit override owns that backend while
// the app-global remote remains the authority for inherited/default profiles.
export function resolveProfileSessionAggregateRoute({
  globalRemote,
  primaryProfileRemoteOverride
}: ProfileSessionAggregateRouteOptions): ProfileSessionAggregateRoute {
  return globalRemote && primaryProfileRemoteOverride ? 'global-remote' : 'primary'
}

/** Fetch the primary backend's profile-aware session slice, falling back to an empty result when unavailable. */
export async function fetchPrimaryProfileSessions(
  searchParams: URLSearchParams,
  fetchJsonForProfile: FetchJsonForProfile,
  aggregateOptions?: ProfileSessionAggregateFetchOptions
): Promise<ProfileSessionsResponse> {
  const fetchJson =
    aggregateOptions && resolveProfileSessionAggregateRoute(aggregateOptions) === 'global-remote'
      ? aggregateOptions.fetchJsonForGlobalRemote
      : fetchJsonForProfile

  try {
    return (await fetchJson(null, `/api/profiles/sessions?${searchParams}`)) as ProfileSessionsResponse
  } catch {
    return { sessions: [], total: 0, profile_totals: {} }
  }
}

/** One CONNECTED (already-pooled) registry gateway whose sessions belong in the
 *  unified list. `backends` carries the resolved descriptors: one per pooled
 *  (connection, profile) pair for ssh-scoped sources, a single shared host for
 *  remote/cloud. The caller resolves descriptors — this module only fetches,
 *  tags, and dedupes, so it stays unit-testable. */
export interface RegistrySessionSource {
  connectionId: string
  /** 'ssh' backends each run AS one remote profile; anything else is a shared
   *  host serving every profile via ?profile=. */
  kind: string
  backends: Array<{ descriptor: unknown; profileLabel: null | string }>
}

type GetJsonForDescriptor = (descriptor: unknown, path: string) => Promise<unknown>

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

  const tag = (data: unknown, connectionId: string, profileLabel: null | string) => {
    for (const row of rowsOf(data)) {
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
      if (source.kind === 'ssh') {
        // Each ssh-scoped backend serves its own state.db natively.
        await Promise.all(
          source.backends.map(async ({ descriptor, profileLabel }) => {
            const params = new URLSearchParams(searchParams)
            params.delete('profile')

            const data = await getJson(descriptor, `/api/sessions?${params}`).catch(() => null)

            if (data) {
              tag(data, source.connectionId, profileLabel || 'default')
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

      let data = await getJson(shared.descriptor, `/api/profiles/sessions?${params}`).catch(() => null)

      if (!data) {
        // Older remote without the aggregator: its own default-profile list.
        const flat = new URLSearchParams(searchParams)
        flat.delete('profile')
        data = await getJson(shared.descriptor, `/api/sessions?${flat}`).catch(() => null)
      }

      if (data) {
        tag(data, source.connectionId, null)
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

export async function fetchRemoteProfileSessions(
  profile: string,
  searchParams: URLSearchParams,
  fetchJsonForProfile: FetchJsonForProfile
): Promise<SessionListResponse> {
  const params = new URLSearchParams(searchParams)
  params.delete('profile') // the remote serves its own database

  const requestedLimit = Number(params.get('limit'))
  const requestedOffset = Number(params.get('offset') || '0')

  const needsPaging =
    Number.isInteger(requestedLimit) &&
    requestedLimit > REMOTE_SESSION_PAGE_LIMIT &&
    Number.isInteger(requestedOffset) &&
    requestedOffset >= 0

  if (!needsPaging) {
    return (await fetchJsonForProfile(profile, `/api/sessions?${params}`)) as SessionListResponse
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

    const page = (await fetchJsonForProfile(profile, `/api/sessions?${pageParams}`)) as SessionListResponse
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
