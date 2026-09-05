import { isMissingRestEndpoint } from '@/lib/gateway-rpc'
import { maybeBackfillLegacySessionOwners } from '@/lib/legacy-session-owner-backfill'
import { stampRowsWithOwningConnection } from '@/lib/session-owner-stamp'
import { recordTranscriptTail } from '@/store/transcript-tail'
import type {
  PaginatedSessions,
  SessionInfo,
  SessionMessage,
  SessionMessagesResponse,
  SessionSearchResponse
} from '@/types/hermes'

import { capabilityScoped, getApiRequestConnection, hermesApi, type ProfileScope, profileScoped } from './client'

const SESSION_LIST_REQUEST_TIMEOUT_MS = 60_000

function sessionScoped(scope?: ProfileScope): { connectionId?: string; profile?: string } {
  if (scope === undefined || scope === null) {
    return {}
  }

  const scoped = capabilityScoped(scope)

  if (typeof scope === 'object' && scope.connectionId?.trim() === 'local') {
    return { ...scoped, connectionId: 'local' }
  }

  return scoped
}

function sessionScopeQuery(scope?: ProfileScope): string {
  const profile = sessionScoped(scope).profile

  return profile ? `?profile=${encodeURIComponent(profile)}` : ''
}

/**
 * The active registered gateway owns every row it returns, but its HTTP APIs
 * correctly know nothing about this Desktop-local registry id. Preserve an
 * explicit owner from a multi-source response; otherwise stamp the active
 * non-local source so a later resume cannot fall back to a same-named local
 * profile. Delegates to the canonical row-stamp helper so this stays the ONE
 * write shape for connection_id on backend-returned rows.
 */
function stampActiveConnectionOwner(sessions: SessionInfo[]): SessionInfo[] {
  // Durable half of the same ownership contract (#94724): enumeration under
  // registry topology triggers the one-shot server-side owner backfill for
  // the serving store when its owner is a single match. Fire-and-forget;
  // idempotent server-side; never blocks or fails the list that triggered it.
  maybeBackfillLegacySessionOwners()

  return stampRowsWithOwningConnection(sessions, getApiRequestConnection())
}

/**
 * Trim a page to its window WITHOUT discarding pinned rows.
 *
 * The list endpoints deliberately back-fill pinned conversations past their
 * LIMIT — a pin means "always reachable", so an aged-out pinned chat is
 * appended after the recency window. A plain `slice(0, limit)` throws exactly
 * those rows away again, which is why pins silently stopped rendering past
 * some count: the sidebar could only ever show the pins that happened to fall
 * inside the most-recent page.
 */
function pageWindow(sessions: SessionInfo[], limit: number): SessionInfo[] {
  if (sessions.length <= limit) {
    return sessions
  }

  const recent = sessions.slice(0, limit)

  return [...recent, ...sessions.slice(limit).filter(session => session.pinned)]
}

export async function listSessions(
  limit = 40,
  minMessages = 0,
  archived: 'exclude' | 'include' | 'only' = 'exclude',
  order: 'created' | 'recent' = 'recent'
): Promise<PaginatedSessions> {
  const result = await hermesApi<PaginatedSessions>({
    ...profileScoped(),
    path:
      `/api/sessions?limit=${limit}&offset=0&min_messages=${Math.max(0, minMessages)}` +
      `&archived=${archived}&order=${order}`,
    timeoutMs: SESSION_LIST_REQUEST_TIMEOUT_MS
  })

  return {
    ...result,
    sessions: pageWindow(stampActiveConnectionOwner(result.sessions), limit),
    offset: 0
  }
}

// Unified, read-only session list aggregated across ALL profiles. Served by the
// primary backend straight off each profile's state.db — no per-profile backend
// is spawned. Single-profile users get the same rows as listSessions(), tagged
// profile="default".
// Source scoping lets callers split the unified list into independent slices:
// recents pass `excludeSources: ['cron']`, the cron-jobs section passes
// `source: 'cron'`. Without this a burst of (always-newest) cron sessions
// consumes the whole recents page and starves real conversations.
export interface SessionSourceFilter {
  source?: string
  excludeSources?: string[]
}

export async function listAllProfileSessions(
  limit = 40,
  minMessages = 0,
  archived: 'exclude' | 'include' | 'only' = 'exclude',
  order: 'created' | 'recent' = 'recent',
  profile: 'all' | (string & {}) = 'all',
  filter: SessionSourceFilter = {}
): Promise<PaginatedSessions> {
  const sourceParam = filter.source ? `&source=${encodeURIComponent(filter.source)}` : ''

  const excludeParam = filter.excludeSources?.length
    ? `&exclude_sources=${encodeURIComponent(filter.excludeSources.join(','))}`
    : ''

  const result = await hermesApi<PaginatedSessions>({
    ...profileScoped(),
    path:
      `/api/profiles/sessions?limit=${limit}&offset=0&min_messages=${Math.max(0, minMessages)}` +
      `&archived=${archived}&order=${order}&profile=${encodeURIComponent(profile)}${sourceParam}${excludeParam}`,
    timeoutMs: SESSION_LIST_REQUEST_TIMEOUT_MS
  })

  return {
    ...result,
    sessions: pageWindow(stampActiveConnectionOwner(result.sessions), limit),
    offset: 0
  }
}

// Batched sidebar slices in one request: recents (scoped to the active profile),
// cron, and messaging. The backend opens each profile's state.db once and runs
// all three filtered queries, replacing three separate listAllProfileSessions
// calls that each reopened + re-counted every profile DB per refresh. Electron
// splices remote profiles per slice (see interceptSessionRequestForRemote).
export interface SidebarSessionSlice {
  sessions: SessionInfo[]
  /** Per-profile "the window came back full, more rows exist on disk" flags —
   *  what pagination needs, without a COUNT(*) per profile DB per refresh. */
  profiles_truncated?: Record<string, boolean>
  /** Per-profile tokens and spend over every session, not just this window.
   *  Absent from the legacy per-slice endpoint, which has no aggregate. */
  profiles_usage?: Record<string, { cost_usd: number; tokens: number }>
  /** Profiles whose scan for THIS slice failed. Batched `/sidebar` stamps the
   *  same profile errors on every slice (one DB open). Legacy per-slice calls
   *  stamp only the slice that actually failed, so a cron I/O error cannot
   *  carry-forward recents. */
  errors?: Array<{ profile: string; error: string }>
}

/** Which profiles filled their per-profile window in a returned page. The
 *  legacy per-slice endpoint doesn't report this, so derive it from the rows:
 *  a profile at (or over) the cap still has more on disk. Pinned rows are
 *  discounted — they're back-filled past the LIMIT, so counting them fakes a
 *  full page and leaves a "Load more" that can never resolve. */
function profilesTruncatedFrom(sessions: SessionInfo[], cap: number): Record<string, boolean> {
  const counts = new Map<string, number>()

  for (const session of sessions) {
    const key = session.profile || 'default'

    counts.set(key, (counts.get(key) ?? 0) + (session.pinned ? 0 : 1))
  }

  return Object.fromEntries([...counts].map(([name, count]) => [name, count >= cap]))
}

export interface SidebarSessionsResponse {
  recents: SidebarSessionSlice
  cron: SidebarSessionSlice
  messaging: SidebarSessionSlice
  errors?: Array<{ profile: string; error: string }>
}

export interface SidebarSessionsRequest {
  recentsProfile: 'all' | (string & {})
  recentsLimit: number
  recentsExclude: string[]
  cronLimit: number
  messagingLimit: number
  messagingExclude: string[]
}

// The batched /sidebar endpoint shipped later than the per-slice route, so a
// newer desktop can meet an older backend that 404s it ("No such API
// endpoint"). Endpoint-missing is a capability signal, not a transient
// failure: remember it (per renderer lifetime — a runtime home change reloads
// the window and re-probes) and serve every subsequent refresh straight from
// the three proven per-slice calls instead of re-probing a known-dead route
// once per turn/broadcast.
let sidebarBatchEndpointMissing = false

// Capability flags are per-backend facts. A hard re-home reloads the window
// (module state resets naturally), but a soft gateway switch re-dials in
// place — the next backend may well have the batched route, so the switch
// paths call this to re-probe rather than leak the old backend's capability.
export function resetSidebarBatchCapability() {
  sidebarBatchEndpointMissing = false
}

// Compatibility fallback: reassemble the three sidebar slices from the
// per-slice endpoint, mirroring the batched route's semantics (min_messages=1,
// archived excluded, recency order; every slice scoped to the caller's profile).
// Rides the same Electron remote-splice
// interception as the pre-batching desktop, so remote profiles stay correct.
async function listSidebarSessionsLegacy(req: SidebarSessionsRequest): Promise<SidebarSessionsResponse> {
  const [recents, cron, messaging] = await Promise.all([
    listAllProfileSessions(req.recentsLimit, 1, 'exclude', 'recent', req.recentsProfile, {
      excludeSources: req.recentsExclude
    }),
    listAllProfileSessions(req.cronLimit, 1, 'exclude', 'recent', req.recentsProfile, { source: 'cron' }),
    listAllProfileSessions(req.messagingLimit, 1, 'exclude', 'recent', req.recentsProfile, {
      excludeSources: req.messagingExclude
    })
  ])

  const recentsErrors = recents.errors ?? []
  const cronErrors = cron.errors ?? []
  const messagingErrors = messaging.errors ?? []

  return {
    recents: {
      profiles_truncated: profilesTruncatedFrom(recents.sessions, req.recentsLimit),
      sessions: recents.sessions,
      ...(recentsErrors.length ? { errors: recentsErrors } : {})
    },
    cron: {
      sessions: cron.sessions,
      ...(cronErrors.length ? { errors: cronErrors } : {})
    },
    messaging: {
      sessions: messaging.sessions,
      ...(messagingErrors.length ? { errors: messagingErrors } : {})
    }
  }
}

/** The PR each of these sessions opened, recovered from its own transcript —
 *  for sessions whose recorded branch can't answer (they started on trunk and
 *  did the work in a worktree). Also returns every id it looked at, so the
 *  caller can remember a miss and never ask again. */
export function scanSessionPullRequests(
  ids: string[]
): Promise<{ pull_requests: Record<string, { number: number; url: string }>; scanned: string[] }> {
  return hermesApi<{
    pull_requests: Record<string, { number: number; url: string }>
    scanned: string[]
  }>({
    path: '/api/profiles/sessions/pull-requests',
    method: 'POST',
    body: { ids }
  })
}

export async function listSidebarSessions(req: SidebarSessionsRequest): Promise<SidebarSessionsResponse> {
  if (sidebarBatchEndpointMissing) {
    return listSidebarSessionsLegacy(req)
  }

  const params = new URLSearchParams({
    recents_profile: req.recentsProfile,
    recents_limit: String(Math.max(1, req.recentsLimit)),
    cron_limit: String(Math.max(1, req.cronLimit)),
    messaging_limit: String(Math.max(1, req.messagingLimit))
  })

  if (req.recentsExclude.length) {
    params.set('recents_exclude', req.recentsExclude.join(','))
  }

  if (req.messagingExclude.length) {
    params.set('messaging_exclude', req.messagingExclude.join(','))
  }

  let result: SidebarSessionsResponse

  try {
    result = await hermesApi<SidebarSessionsResponse>({
      ...profileScoped(),
      path: `/api/profiles/sessions/sidebar?${params.toString()}`,
      timeoutMs: SESSION_LIST_REQUEST_TIMEOUT_MS
    })
  } catch (err) {
    // Safe to read a 404 as route-missing here: this GET has no path params,
    // so it cannot 404 on a bad id.
    if (!isMissingRestEndpoint(err)) {
      throw err
    }

    // Older backend without the batched route (desktop/runtime version skew).
    sidebarBatchEndpointMissing = true

    return listSidebarSessionsLegacy(req)
  }

  return {
    recents: {
      ...result.recents,
      sessions: stampActiveConnectionOwner(result.recents?.sessions ?? []),
      ...(result.errors?.length ? { errors: result.errors } : {})
    },
    cron: {
      ...result.cron,
      sessions: stampActiveConnectionOwner(result.cron?.sessions ?? []),
      ...(result.errors?.length ? { errors: result.errors } : {})
    },
    messaging: {
      ...result.messaging,
      sessions: stampActiveConnectionOwner(result.messaging?.sessions ?? []),
      ...(result.errors?.length ? { errors: result.errors } : {})
    },
    errors: result.errors
  }
}

// Mutations take the owning `profile` so Electron can route them to the correct
// remote backend or local profile scope. Omit for the current/default profile.
export function setSessionArchived(id: string, archived: boolean, profile?: string | null): Promise<{ ok: boolean }> {
  // Carry the owning profile IN THE PATCH BODY, mirroring renameSession — the
  // backend reads its target DB from body.profile (_open_session_db_for_profile).
  // Passing it only as request.profile (Electron routing) is not enough on a
  // remote gateway with no remoteProfile alias: the archive lands on the wrong
  // (default) state.db, no-ops on a missing row, and the archived/unarchived
  // state silently fails to stick — the same class as the unscoped DELETE.
  return hermesApi<{ ok: boolean }>({
    ...(profile ? { profile } : {}),
    path: `/api/sessions/${encodeURIComponent(id)}`,
    method: 'PATCH',
    body: { archived, ...(profile ? { profile } : {}) }
  })
}

// Mirror a sidebar pin to the backend "keep" flag so the sessions.auto_archive
// sweep (which runs backend-side, blind to Desktop localStorage) never hides a
// pinned chat. Best-effort: the sidebar stays localStorage-driven for its own
// display; this only feeds the backend policy.
export function setSessionPinnedRemote(id: string, pinned: boolean, profile?: string | null): Promise<{ ok: boolean }> {
  // Owning profile in the PATCH body (see setSessionArchived / renameSession):
  // the handler reads its target DB from body.profile, so a remote/foreign
  // profile's pin must travel in the body or it no-ops on the wrong state.db.
  return hermesApi<{ ok: boolean }>({
    ...(profile ? { profile } : {}),
    path: `/api/sessions/${encodeURIComponent(id)}`,
    method: 'PATCH',
    body: { pinned, ...(profile ? { profile } : {}) }
  })
}

// Mirror a sidebar unread toggle to the backend read-state watermark
// (sessions.last_read_at via SessionDB.set_session_read). Same profile
// routing as the other session mutations: a remote session's row lives only
// on its remote host, so the owning profile must travel with the request.
export function setSessionUnreadRemote(id: string, unread: boolean, profile?: string | null): Promise<{ ok: boolean }> {
  // Owning profile in the PATCH body (see setSessionArchived / renameSession):
  // the handler reads its target DB from body.profile, so a remote/foreign
  // profile's unread toggle must travel in the body or it no-ops on the wrong
  // state.db.
  return hermesApi<{ ok: boolean }>({
    ...(profile ? { profile } : {}),
    path: `/api/sessions/${encodeURIComponent(id)}`,
    method: 'PATCH',
    body: { unread, ...(profile ? { profile } : {}) }
  })
}

export function searchSessions(query: string): Promise<SessionSearchResponse> {
  return hermesApi<SessionSearchResponse>({
    path: `/api/sessions/search?q=${encodeURIComponent(query)}`
  })
}

// Resolves a single session row by id on one backend (the active profile, or
// the given `profile`). The backend resolves exact ids and unique prefixes and
// 404s when the id isn't on that profile — so a cheap by-id lookup replaces the
// cross-profile list scan when locating an unknown id's owner.
export function getSession(id: string, profile?: ProfileScope): Promise<SessionInfo> {
  const suffix = sessionScopeQuery(profile)

  return hermesApi<SessionInfo>({
    ...sessionScoped(profile),
    path: `/api/sessions/${encodeURIComponent(id)}${suffix}`
  })
}

// Reads another profile's transcript. For a remote profile Electron reroutes
// this GET to the remote backend (which serves its own state.db); for a local
// profile the primary opens that profile's state.db via ?profile=. Omit for
// the current/default profile.
export function getSessionMessages(
  id: string,
  profile?: ProfileScope,
  page: { limit?: number; offset?: number; order?: 'latest' | 'oldest'; includeCompacted?: boolean } = {},
  options: { passive?: boolean } = {}
): Promise<SessionMessagesResponse> {
  const query = new URLSearchParams()

  const sessionScope = sessionScoped(profile)

  if (sessionScope.profile) {
    query.set('profile', sessionScope.profile)
  }

  if (page.limit !== undefined) {
    query.set('limit', String(page.limit))
  }

  if (page.offset !== undefined) {
    query.set('offset', String(page.offset))
  }

  if (page.order) {
    query.set('order', page.order)
  }

  if (page.includeCompacted !== undefined) {
    query.set('include_compacted', String(page.includeCompacted))
  }

  const suffix = query.size ? `?${query.toString()}` : ''

  return hermesApi<SessionMessagesResponse>({
    ...sessionScope,
    path: `/api/sessions/${encodeURIComponent(id)}/messages${suffix}`,
    ...(options.passive ? { passive: true } : {})
  })
}

/**
 * The initial hydration page: enough tail to fill the transcript window a few
 * times over, small enough that opening a long session doesn't ship (and
 * convert) hundreds of rows nobody has scrolled to. Older rows load on demand
 * via `getOlderSessionMessages` when "Show earlier" exhausts the in-memory
 * store (see app/chat/transcript-backfill).
 */
export const LATEST_SESSION_MESSAGES_LIMIT = 120

export function getLatestSessionMessages(id: string, profile?: ProfileScope, passive = false): Promise<SessionMessagesResponse> {
  // includeCompacted: durable display history must include rows preserved by
  // in-place compaction (active=0, compacted=1); without them the transcript
  // silently ends at the compaction boundary and earlier turns are unreachable.
  return getSessionMessages(
    id,
    profile,
    {
      limit: LATEST_SESSION_MESSAGES_LIMIT,
      order: 'latest',
      includeCompacted: true
    },
    passive ? { passive: true } : {}
  ).then(page => {
    // Record whether the tail was truncated (page came back full) and where
    // the next older page starts, so "Show earlier" can backfill over REST
    // (app/chat/transcript-backfill). Keyed under both the requested id and
    // the resolved id — callers hold either.
    recordTranscriptTail(id, page, profile)

    if (page.session_id && page.session_id !== id) {
      recordTranscriptTail(page.session_id, page, profile)
    }

    return page
  })
}

/**
 * READ-ONLY stored-transcript lookup that never routes a live session
 * (#94724 no-owner recovery). Tries the ambient/primary store first, then
 * probes every registered NON-local connection by id — a REST read of a
 * backend's own state.db is side-effect free (a miss is a plain 404, no
 * session is minted or resumed anywhere), so probing across backends is safe
 * where live routing would be a guess. Returns null when no reachable
 * backend holds the transcript.
 */
export async function fetchStoredTranscriptAcrossBackends(id: string): Promise<SessionMessagesResponse | null> {
  try {
    return await getLatestSessionMessages(id)
  } catch {
    // Not on the ambient store — probe the registered backends below.
  }

  const { $connectionsRegistry } = await import('@/store/connection-registry-state')

  const connections = ($connectionsRegistry.get()?.connections ?? []) as Array<{ id?: string }>

  for (const connection of connections) {
    const connectionId = connection.id?.trim()

    if (!connectionId || connectionId === 'local' || connectionId === getApiRequestConnection()) {
      continue
    }

    try {
      return await getLatestSessionMessages(id, { connectionId, profile: 'default' })
    } catch {
      // Not on this backend (or it is unreachable); try the next.
    }
  }

  return null
}

/**
 * One page of messages OLDER than the `offset` newest rows.
 *
 * Backend semantics (`_handle_session_messages` → `SessionDB.get_messages`
 * with `latest=True`): the offset is measured back from the NEWEST message
 * and the selected page is returned in chronological order. So after a tail
 * hydration of N rows, `getOlderSessionMessages(id, profile, N)` returns the
 * page immediately preceding it, ready to prepend.
 *
 * Legacy backends without pagination support return the full transcript and
 * no `pagination` metadata — callers detect that via the missing field and
 * treat the response as the complete history (see transcript-backfill).
 */
export function getOlderSessionMessages(
  id: string,
  profile: ProfileScope,
  offset: number,
  limit: number = LATEST_SESSION_MESSAGES_LIMIT
): Promise<SessionMessagesResponse> {
  return getSessionMessages(id, profile, { includeCompacted: true, limit, offset, order: 'latest' })
}

export async function getAllSessionMessages(
  id: string,
  profile?: ProfileScope,
  options: { maxJsonChars?: number } = {}
): Promise<SessionMessagesResponse> {
  const messages: SessionMessage[] = []
  const pageSize = 500
  const maxJsonChars = options.maxJsonChars ?? 32_000_000
  let jsonChars = 0
  let offset = 0
  let resolvedSessionId = id

  while (true) {
    const page = await getSessionMessages(id, profile, {
      limit: pageSize,
      offset,
      order: 'oldest',
      includeCompacted: true
    })

    resolvedSessionId = page.session_id
    jsonChars += (JSON.stringify(page.messages) ?? '').length

    if (jsonChars > maxJsonChars) {
      throw new Error(
        'Session transcript exceeds the Desktop safe-load limit; use the Web Dashboard export for this session.'
      )
    }

    messages.push(...page.messages)

    // Legacy backends ignore pagination and return the full transcript.
    if (!page.pagination || page.messages.length === 0 || page.messages.length < page.pagination.limit) {
      break
    }

    offset += page.messages.length
  }

  return { session_id: resolvedSessionId, messages }
}

export function deleteSession(id: string, profile?: ProfileScope): Promise<{ ok: boolean }> {
  // Scope the DELETE to the owning profile IN THE URL, mirroring getSession /
  // getSessionMessages. Passing the profile only via request.profile (which the
  // Electron main process consumes for backend routing) is NOT enough: on a
  // remote gateway whose connection has no remoteProfile alias, the main-process
  // path rewrite leaves the URL unscoped, so the backend opens its OWN (default)
  // state.db, fails to find another profile's session, and returns
  // {ok:true, already_absent:true}. The row then vanishes optimistically but was
  // never deleted and reappears on the next sidebar refresh — the "All Profiles
  // delete doesn't stick" report. The endpoint already honours ?profile=
  // (get/rename/messages all send it); DELETE was the one mutation that dropped
  // it after the api/ split. request.profile stays on for per-profile remote
  // override + global-remote routing (both re-read/re-append the param).
  const suffix = sessionScopeQuery(profile)

  return hermesApi<{ ok: boolean }>({
    ...sessionScoped(profile),
    path: `/api/sessions/${encodeURIComponent(id)}${suffix}`,
    method: 'DELETE'
  })
}

export function renameSession(
  id: string,
  title: string,
  profile?: string | null
): Promise<{ ok: boolean; title: string }> {
  return hermesApi<{ ok: boolean; title: string }>({
    ...(profile ? { profile } : {}),
    path: `/api/sessions/${encodeURIComponent(id)}`,
    method: 'PATCH',
    body: { title, ...(profile ? { profile } : {}) }
  })
}
