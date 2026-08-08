export interface ProfileSessionsResponse {
  sessions: unknown[]
  total: number
  profile_totals: Record<string, number>
  [key: string]: unknown
}

type FetchJsonForProfile = (profile: string | null, path: string) => Promise<unknown>

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
