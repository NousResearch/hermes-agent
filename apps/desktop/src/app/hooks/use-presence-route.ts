import { useEffect } from 'react'
import { useLocation } from 'react-router'

import { refreshWindowPresence } from '@/store/presence'

/**
 * Tell this window's presence reporter the moment the route moves. The route decides which
 * conversation is in view, and HashRouter moves it with `history.pushState`, which fires no
 * `hashchange` event the reporter could hear.
 */
export function usePresenceRoute(): void {
  const { pathname, search } = useLocation()

  useEffect(() => {
    refreshWindowPresence()
  }, [pathname, search])
}
