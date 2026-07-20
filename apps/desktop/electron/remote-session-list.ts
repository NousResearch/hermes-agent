interface SessionListResponse {
  limit?: number
  offset?: number
  sessions?: unknown
  total?: number
  [key: string]: unknown
}

/**
 * Adapt one remote backend's `/api/sessions` response to the
 * `/api/profiles/sessions` contract consumed by Desktop.
 *
 * The single-profile endpoint does not know the Desktop profile name and does
 * not return `profile_totals`. The latter is pagination metadata: without it,
 * a profile-scoped sidebar treats the first page as the complete result set.
 */
export function normalizeRemoteSessionList(profile: string, data: unknown) {
  const source = data && typeof data === 'object' ? (data as SessionListResponse) : {}
  const rawSessions = Array.isArray(source.sessions) ? source.sessions : []

  const sessions = rawSessions.map(session =>
    session && typeof session === 'object'
      ? { ...session, profile, is_default_profile: false }
      : session
  )

  const total = Number.isFinite(source.total) ? Number(source.total) : sessions.length

  return {
    ...source,
    sessions,
    total,
    profile_totals: { [profile]: total }
  }
}
