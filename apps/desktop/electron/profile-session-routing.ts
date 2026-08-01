export interface ProfileSessionsResponse {
  sessions: unknown[]
  total: number
  profile_totals: Record<string, number>
  [key: string]: unknown
}

type FetchJsonForProfile = (profile: string | null, path: string) => Promise<unknown>

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
