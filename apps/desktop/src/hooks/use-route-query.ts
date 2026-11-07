import { useMemo, useRef } from 'react'
import { type Location, type NavigateFunction, useLocation, useNavigate } from 'react-router'

/** A page's deep-link query, read off the route that is showing the page. */
export interface RouteQuery {
  /** The query string of the navigation showing the page. */
  params: URLSearchParams
  /**
   * Act on this link. True for the first mounted copy of the page that asks
   * (and again for that same copy), false for every other copy, so a page open
   * in the workspace AND in a split tile acts once. The query is dropped from
   * the URL with a history replace: Back does not replay it, and sending the
   * same link again is a fresh navigation.
   */
  take: () => boolean
}

// react-router's hooks throw with no router context (a page rendered inside a
// plugin dialog, a bare test render). Router presence is stable for a mounted
// instance's lifetime, so the try/catch never changes the hook count between
// renders — the same guard as Settings' useDeepLinkHighlight.
function useOptionalLocation(): Location | null {
  try {
    return useLocation()
  } catch {
    return null
  }
}

function useOptionalNavigate(): NavigateFunction | null {
  try {
    return useNavigate()
  } catch {
    return null
  }
}

// Which mounted copy took each navigation. The router mints a new location
// object per navigation, so a weak key needs no cleanup and never outlives it.
const takenBy = new WeakMap<Location, object>()

/**
 * The current route's query while the app is ON `path` (a contributed page's
 * own path, e.g. `/kanban`); null on any other route, outside the router, or
 * when the route carries no query. Routes have no path params: a page that
 * accepts a deep link (`host.navigate('/my-page?item=x')`) reads it here,
 * acts on it once via `take()`, and keeps whatever it needs in its own state.
 */
export function useRouteQuery(path: string): null | RouteQuery {
  const location = useOptionalLocation()
  const navigate = useOptionalNavigate()
  const owner = useRef<object>({})

  return useMemo(() => {
    if (!location || !navigate || location.pathname !== path || !location.search) {
      return null
    }

    return {
      params: new URLSearchParams(location.search),
      take: () => {
        const holder = takenBy.get(location)

        if (holder) {
          return holder === owner.current
        }

        takenBy.set(location, owner.current)
        navigate({ hash: location.hash, pathname: location.pathname }, { replace: true })

        return true
      }
    }
  }, [location, navigate, path])
}
