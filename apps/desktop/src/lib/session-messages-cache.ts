import { getSession, getSessionMessages } from '@/hermes'

import { queryClient } from './query-client'

/** Query key for a session's messages (scoped by profile + session id). */
export function sessionMessagesQueryKey(
  profile: string | null,
  sessionId: string
): readonly [string, string, string | null] {
  return ['session-messages', sessionId, profile]
}

/** Prefetch and cache a session's transcript for instant SWR display. */
export async function prefetchSessionMessages(
  sessionId: string,
  profile: string | null
): Promise<void> {
  // If profile is not provided, try to resolve from the session row
  let resolvedProfile = profile

  if (!resolvedProfile) {
    try {
      const session = await getSession(sessionId)
      resolvedProfile = session.profile ?? null
    } catch {
      // If we can't resolve, use null (default profile)
      resolvedProfile = null
    }
  }

  await queryClient.prefetchQuery({
    queryKey: sessionMessagesQueryKey(resolvedProfile, sessionId),
    queryFn: () => getSessionMessages(sessionId, resolvedProfile),
    staleTime: 30_000,
    gcTime: 5 * 60_000,
  })
}

/** Prime the cache with existing data (used when navigating to a session we already have). */
export function primeSessionMessagesCache(
  sessionId: string,
  profile: string | null,
  messages: unknown
): void {
  queryClient.setQueryData(sessionMessagesQueryKey(profile, sessionId), messages)
}

/** Get cached messages if available (for SWR placeholderData). */
export function getCachedSessionMessages(
  sessionId: string,
  profile: string | null
): unknown | undefined {
  return queryClient.getQueryData(sessionMessagesQueryKey(profile, sessionId))
}

/** Invalidate a session's messages cache (e.g., after a new turn completes). */
export function invalidateSessionMessages(sessionId: string, profile: string | null): void {
  queryClient.invalidateQueries({ queryKey: sessionMessagesQueryKey(profile, sessionId) })
}