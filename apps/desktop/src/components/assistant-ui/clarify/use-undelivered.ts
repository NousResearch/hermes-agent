import { useEffect, useState } from 'react'

import { sessionClarifyRequest } from '@/store/clarify'
import { $gateway } from '@/store/gateway'
import { requestForOwnedSession } from '@/store/session-states'

/** How long a painted card may wait for its gateway request before asking the
 *  backend to re-deliver it. `clarify.request` normally trails `tool.start` by
 *  a tick; seconds of silence mean the frame was lost on the way. */
const CLARIFY_DELIVERY_GRACE_MS = 4_000

/**
 * True once the card has waited past the grace period for a request that
 * never arrived, and a re-delivery attempt found nothing (#98645).
 *
 * The attempt asks the owner socket for `session.events.since` from the end
 * of the ring: no events come back, but the channel re-delivers the
 * session's `open_requests` to the request handlers before the call
 * resolves, so a request the backend still holds parks and the card goes
 * live on its own.
 */
export function useUndeliveredClarify(sessionId: null | string, waiting: boolean): boolean {
  const [undelivered, setUndelivered] = useState(false)

  useEffect(() => {
    if (!waiting || !sessionId) {
      return
    }

    let cancelled = false

    const timer = window.setTimeout(async () => {
      const gateway = $gateway.get()

      if (gateway) {
        try {
          await requestForOwnedSession(
            sessionId,
            gateway.request.bind(gateway) as typeof gateway.request,
            'session.events.since',
            {
              last_seen: Number.MAX_SAFE_INTEGER,
              session_id: sessionId
            }
          )
        } catch {
          // An older backend or a dropped socket: the notice is still right.
        }
      }

      if (!cancelled && !sessionClarifyRequest(sessionId).get()) {
        setUndelivered(true)
      }
    }, CLARIFY_DELIVERY_GRACE_MS)

    return () => {
      cancelled = true
      window.clearTimeout(timer)
      setUndelivered(false)
    }
  }, [sessionId, waiting])

  return waiting && undelivered
}
