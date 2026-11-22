import { resolveSessionOwner } from '@/app/session/hooks/use-session-actions/utils'
import type { GatewayFileOrigin } from '@/lib/media'
import { profileScopeForSessionOwner } from '@/store/session-request-router'

/** Origin for a file a chat transcript names: the session AND the profile (or exact owner route)
 *  that owns it. The gateway looks a `session_id` up in the request's profile, so a save naming the
 *  session without its owner 404s ("Session not found") whenever the chat lives on a profile other
 *  than the window's. A chat not stored yet names its live runtime session; no session, no origin. */
export async function sessionFileOrigin(
  storedSessionId: null | string,
  runtimeId: null | string
): Promise<GatewayFileOrigin | undefined> {
  if (!storedSessionId) {
    return runtimeId ? { sessionId: runtimeId } : undefined
  }

  const scope = profileScopeForSessionOwner(await resolveSessionOwner(storedSessionId))

  return {
    sessionId: storedSessionId,
    ...(typeof scope === 'string' ? { profile: scope } : scope ? { owner: scope } : {})
  }
}
