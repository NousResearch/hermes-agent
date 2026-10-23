import { getSessionOwnerHint } from '@/store/session'
import type { SessionOwnerRoute } from '@/store/session-request-router'

/** The main pane's owner route for a stored session.
 *
 * The ambient-keyed hint first — the record an open under the ACTIVE profile
 * leaves behind — then the session's single filed hint. The second rung is the
 * in-place Bot chat: `host.openSession` files its hint under the BOT's own
 * `(connection, profile)` key while chrome stays on the launch profile, so the
 * ambient-keyed lookup misses and the pane used to fall back to the ambient
 * profile — live voice then spoke with the default profile's voice instead of
 * the Bot's (#137801). A session with hints filed under several profiles
 * stays unowned (the ambiguous case is not the Bot chat's). */
export function paneOwnerRoute(
  storedId: null | string,
  ambient: null | { connectionId: string; profile: string }
): SessionOwnerRoute | undefined {
  if (!storedId) {
    return undefined
  }

  return (ambient ? getSessionOwnerHint(storedId, ambient) : undefined) ?? getSessionOwnerHint(storedId)
}
