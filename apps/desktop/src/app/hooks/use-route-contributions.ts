import { useEffect } from 'react'

import { ROUTES_AREA, routeSessionId, syncWorkspaceRoute } from '@/app/routes'
import { useContributions } from '@/contrib/react/use-contributions'

/** The route state that depends on which plugin routes are registered. A plugin
 *  page's path reads as a session id until its route registers, and disk
 *  plugins register after boot, so a reload or deep link on a plugin page is
 *  re-derived when the page registers, not only when the path changes.
 *
 *  Points the workspace at the route (the zone tab bar stands down on a page,
 *  and a page fronts the pane) and returns the stored session id it names. */
export function useRouteContributions(pathname: string): string | null {
  const routeSnapshot = useContributions(ROUTES_AREA)

  useEffect(() => {
    syncWorkspaceRoute(pathname, routeSnapshot)
  }, [pathname, routeSnapshot])

  return routeSessionId(pathname, routeSnapshot)
}
