import { useEffect, useRef } from 'react'

import { $freshSessionRequest } from '@/store/profile'

/**
 * Deliver fresh-session requests synchronously once the listener is mounted.
 *
 * Profile selection bumps the store immediately before it repoints the active
 * gateway. A reactive render/effect bridge is too late here: the new gateway
 * can become active while the foreground refs still identify the old profile's
 * session. Nanostores listeners run inside the originating `.set`, so the
 * foreground teardown completes before gateway activation can continue.
 */
export function useFreshSessionRequests(onRequest: () => void): void {
  const onRequestRef = useRef(onRequest)

  onRequestRef.current = onRequest

  useEffect(() => $freshSessionRequest.listen(() => onRequestRef.current()), [])
}
