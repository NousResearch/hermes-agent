import { useEffect, useState } from 'react'

import { type ConnectionOwner, connectionOwnerFor } from '@/store/connection-request'

/** The gateway route that owns a session's connection operation. Null until it resolves and null when it cannot: a card RPC must reach the
 *  gateway that holds the operation, never whichever one the window happens to have in front. */
export function useConnectionOwner(sessionId: null | string, active: boolean): ConnectionOwner | null {
  const [owner, setOwner] = useState<ConnectionOwner | null>(null)

  useEffect(() => {
    if (!sessionId || !active) {
      setOwner(null)

      return
    }

    let cancelled = false

    void connectionOwnerFor(sessionId, 'connectors.connect').then(resolved => {
      if (!cancelled) {
        setOwner(resolved)
      }
    })

    return () => {
      cancelled = true
    }
  }, [active, sessionId])

  return owner
}
