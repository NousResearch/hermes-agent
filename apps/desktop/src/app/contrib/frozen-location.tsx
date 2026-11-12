/**
 * A subtree that reads a location of its own. react-router 8 has no public API for it: `<Routes
 * location>` scopes only the matching, while `useLocation()`/`useNavigate()`/`useParams()` below
 * still read the router's live location context, and a nested `<Router>` is rejected outright. The
 * kept chat (route-cover.tsx) must not see pages' locations, so this one module touches the router's
 * internal context; frozen-location.test.tsx fails loudly if an upgrade drops or changes it.
 */
import { type ContextType, type ReactNode, useContext } from 'react'
import { UNSAFE_LocationContext as LocationContext } from 'react-router'

export type RouterLocation = ContextType<typeof LocationContext>

/** The router's current location context, to hand to `AtLocation` later. */
export function useRouterLocation(): RouterLocation {
  return useContext(LocationContext)
}

interface AtLocationProps {
  children: ReactNode
  location: RouterLocation
}

/** Renders `children` as if the router were at `location`. */
export function AtLocation({ children, location }: AtLocationProps) {
  return <LocationContext value={location}>{children}</LocationContext>
}
