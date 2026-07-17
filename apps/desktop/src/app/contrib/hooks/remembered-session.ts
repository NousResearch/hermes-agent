import type { SessionInfo } from '@/hermes'

/**
 * Returns the session safe to restore at cold start. Delegate children are
 * transient and deliberately omitted from the sidebar list, so this accepts
 * only metadata fetched directly by id.
 */
export async function resolveRememberedSessionId(
  id: string,
  getSession: (id: string) => Promise<SessionInfo>
): Promise<null | string> {
  const session = await getSession(id)

  return session.source === 'subagent' ? session.parent_session_id ?? null : session.id
}
