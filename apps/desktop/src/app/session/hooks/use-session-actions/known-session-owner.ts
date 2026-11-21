import { resolveSessionRpcOwner } from '@/app/contrib/wiring-routing'
import { getSessionOwnerHint, knownSessionOwner, ownerLookupSessionRows } from '@/store/session'
import type { SessionProfileRoute } from '@/store/session-request-router'
import { runtimeSessionOwner, sessionTileOwnerRoute } from '@/store/session-states'

/** The owner the store already knows for a stored id, without a request: the
 *  sync ladder session RPCs use (tile route, unique hint, tagged row, proven
 *  runtime event). */
export function knownSessionRpcOwner(storedSessionId: string) {
  return resolveSessionRpcOwner({
    routingSessionId: storedSessionId,
    tileOwnerRoute: sessionTileOwnerRoute,
    sessionOwnerHint: getSessionOwnerHint,
    sessionRowOwner: id => knownSessionOwner(ownerLookupSessionRows(), id),
    eventOwner: runtimeSessionOwner
  })
}

/** The exact (connection-bearing) owner the store already knows for a stored
 *  id; undefined when it knows only a bare profile or nothing.
 *
 *  A stored-session probe whose caller names no owner starts here. Its profile
 *  ladder walks the AMBIENT source's profile names, and a session whose known
 *  owner is another connection (This device's Bot Chat on a remote-primary
 *  Desktop) has nothing to find there: every name in that walk reached the
 *  owner's connection through the read pin, which refused or dialed each one
 *  in turn ("Profile \"rift\" no longer exists", and a 120 s remote-dashboard
 *  wait for another). */
export function knownExactOwnerRoute(storedSessionId: string): SessionProfileRoute | undefined {
  const owner = knownSessionRpcOwner(storedSessionId)

  if (!owner || typeof owner !== 'object' || !String(owner.connectionId ?? '').trim()) {
    return undefined
  }

  return owner
}
