import { useCallback, useRef } from 'react'

import { routeSessionId } from '../../routes'

interface RouteLocation {
  hash: string
  pathname: string
  search: string
}

/** The routed stored session and a token for the whole route, each mirrored into a ref so async
 *  work started under one route can tell, when it lands, that the user has since moved on. */
export function useRouteRefs(location: RouteLocation) {
  const routedSessionId = routeSessionId(location.pathname)
  const routedSessionIdRef = useRef(routedSessionId)
  routedSessionIdRef.current = routedSessionId

  const routeTokenRef = useRef('')
  routeTokenRef.current = `${location.pathname}:${location.search}:${location.hash}`

  const getRouteToken = useCallback(() => routeTokenRef.current, [])
  const getRoutedStoredSessionId = useCallback(() => routedSessionIdRef.current, [])

  return { getRouteToken, getRoutedStoredSessionId, routedSessionId, routedSessionIdRef }
}
