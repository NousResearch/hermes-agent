import { normalizeProfileKey } from '@/store/profile-key'
import { parseSessionIdentityKey } from '@/store/session-identity'

// The gateway tags every event — and therefore every native notification —
// with the *runtime* session id (the key under which the session lives in the
// gateway's in-memory `_sessions` map). The chat route, however, is keyed by
// the *stored* session id (`stored_session_id`), which is a different value:
// a brand-new chat gets a runtime id immediately but its stored id is assigned
// when the first turn persists. Navigating to a runtime id therefore tries to
// resume a stored session that does not exist ("session not found") and
// strands the user, who experiences it as the running session being destroyed.
//
// `runtimeIdByStoredSessionId` maps stored -> runtime; this resolves the
// reverse so notification-click navigation lands on the real route. The id is
// returned unchanged when no mapping is known — it may already be a stored id
// (e.g. a notification for a session this window never opened), in which case
// the normal resume/REST lookup handles it.
export function storedSessionIdForNotification(
  profile: string | null | undefined,
  id: string,
  runtimeIdByStoredSessionId: ReadonlyMap<string, string>
): string {
  const profileKey = normalizeProfileKey(profile)

  for (const [storedKey, runtimeId] of runtimeIdByStoredSessionId) {
    const stored = parseSessionIdentityKey(storedKey)

    if (stored?.profile === profileKey && runtimeId === id) {
      return stored.sessionId
    }
  }

  return id
}
