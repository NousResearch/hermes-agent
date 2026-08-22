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

/** Fetch the primary backend's profile-aware session slice, falling back to an empty result when unavailable. */
export async function fetchPrimaryProfileSessions(
  searchParams: URLSearchParams,
  fetchJsonForProfile: FetchJsonForProfile
): Promise<ProfileSessionsResponse> {
  try {
    return (await fetchJsonForProfile(null, `/api/profiles/sessions?${searchParams}`)) as ProfileSessionsResponse
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

/** The cross-profile aggregator caps per-request pages at 500 (it fans the
 *  query out across every profile's state.db — see web_routers/profiles.py). */
const PROFILE_SESSIONS_PAGE_LIMIT = 500

export interface RemoteProfileSessionReadOptions {
  /** The endpoint serves more than one desktop scope (several profiles point
   *  at the same URL — #64999), so it is a multi-profile backend: read THIS
   *  scope through its cross-profile aggregator (?profile=<scope>), where
   *  every row arrives stamped with the real owning profile. That identity is
   *  authoritative and flows through untouched. An older remote without the
   *  aggregator still serves one launch-profile database — that falls back to
   *  the single-profile read AND its scope labeling, exactly as if the
   *  endpoint had never been shared. */
  sharedEndpoint?: boolean
}

type RemoteSessionPageReader = (query: string) => Promise<SessionListResponse>

/**
 * Shared windowing for both remote session endpoints: small requests stay on
 * one call; oversized windows walk limit/offset pages of at most `pageLimit`
 * (each endpoint caps its per-request page size), keeping ordinary rows in
 * order and appending pinned rows that fell outside the window last — the same
 * result one larger request would have produced.
 */
async function pageRemoteSessionWindow(
  params: URLSearchParams,
  pageLimit: number,
  readPage: RemoteSessionPageReader
): Promise<SessionListResponse> {
  const requestedLimit = Number(params.get('limit'))
  const requestedOffset = Number(params.get('offset') || '0')

  const needsPaging =
    Number.isInteger(requestedLimit) &&
    requestedLimit > pageLimit &&
    Number.isInteger(requestedOffset) &&
    requestedOffset >= 0

  if (!needsPaging) {
    return readPage(params.toString())
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
    const pageLimitForCall = Math.min(pageLimit, targetOffset - pageOffset)
    pageParams.set('limit', String(pageLimitForCall))
    pageParams.set('offset', String(pageOffset))

    const page = await readPage(pageParams.toString())
    firstPage ??= page

    const total = nonNegativeNumber(page.total)
    const pageRows = rowsOf(page)

    const windowedCount =
      total !== null
        ? Math.min(pageLimitForCall, Math.max(0, total - pageOffset))
        : Math.min(pageLimitForCall, pageRows.length)

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

    pageOffset += pageLimitForCall
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

/** One remote's OWN database (genuinely single-profile endpoint): strip the
 *  profile param — it serves its state.db natively — and page at its API cap.
 *  Rows are returned as-is; labeling them with the selecting scope is the
 *  caller's call (main.ts does it for exactly this path). */
async function fetchOwnDatabaseRemoteSessions(
  profile: string,
  searchParams: URLSearchParams,
  fetchJsonForProfile: FetchJsonForProfile
): Promise<SessionListResponse> {
  const params = new URLSearchParams(searchParams)
  params.delete('profile') // the remote serves its own database

  return pageRemoteSessionWindow(params, REMOTE_SESSION_PAGE_LIMIT, query =>
    fetchJsonForProfile(profile, `/api/sessions?${query}`) as Promise<SessionListResponse>
  )
}

export async function fetchRemoteProfileSessions(
  profile: string,
  searchParams: URLSearchParams,
  fetchJsonForProfile: FetchJsonForProfile,
  options: RemoteProfileSessionReadOptions = {}
): Promise<SessionListResponse> {
  if (!options.sharedEndpoint) {
    return fetchOwnDatabaseRemoteSessions(profile, searchParams, fetchJsonForProfile)
  }

  try {
    // Multi-profile backend shared by several desktop scopes (#64999): ask its
    // cross-profile aggregator for THIS scope's slice. Every row comes stamped
    // by the backend with its real owning profile — that identity must reach
    // the renderer untouched.
    const params = new URLSearchParams(searchParams)
    params.set('profile', profile)

    return await pageRemoteSessionWindow(params, PROFILE_SESSIONS_PAGE_LIMIT, query =>
      fetchJsonForProfile(profile, `/api/profiles/sessions?${query}`) as Promise<SessionListResponse>
    )
  } catch {
    // Older remote without the aggregator: it still serves ONE launch-profile
    // database whose rows cannot prove which remote profile they belong to —
    // fall back to the single-profile read and its scope labeling.
    const data = await fetchOwnDatabaseRemoteSessions(profile, searchParams, fetchJsonForProfile)

    for (const s of rowsOf(data)) {
      const session = s as Record<string, unknown>
      session.profile = profile
      session.is_default_profile = false
    }

    return data
  }
}
