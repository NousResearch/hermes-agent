/**
 * react-router's `useNavigate()` returns a new function on every location change (it closes over the
 * current pathname to resolve relative targets), so every component that only SENDS the user
 * somewhere re-rendered on each navigation: the whole sidebar (~1,100 components for the agent
 * list alone) twice per rail click. The shell navigates from outside `<Routes>`, where a relative
 * target resolves against the root either way, so one navigate captured where the shell already
 * follows the location serves every such caller and never changes identity.
 */
import { createContext, type ReactNode, useContext, useLayoutEffect, useRef, useState } from 'react'
import { type NavigateFunction, type NavigateOptions, type To, useNavigate } from 'react-router'

export const StableNavigateContext = createContext<NavigateFunction | null>(null)

/** For the one component that provides it: a navigate whose identity never changes, delegating to
 *  the router's latest one. Its caller re-renders on navigation (it reads the router). */
export function useStableNavigateSource(): NavigateFunction {
  const navigate = useNavigate()
  const latest = useRef(navigate)

  useLayoutEffect(() => {
    latest.current = navigate
  }, [navigate])

  const [stable] = useState<NavigateFunction>(
    () =>
      ((to: number | To, options?: NavigateOptions) =>
        typeof to === 'number' ? latest.current(to) : latest.current(to, options)) as NavigateFunction
  )

  return stable
}

/** Navigate without following the location: for shell components (sidebar rows, rails, bars) that
 *  must not re-render when the route changes. Provided by `ContribWiring`. */
export function useStableNavigate(): NavigateFunction {
  const navigate = useContext(StableNavigateContext)

  if (!navigate) {
    throw new Error('useStableNavigate() needs a StableNavigateContext provider (ContribWiring)')
  }

  return navigate
}

/** Provides `useStableNavigate()` to a subtree that is not under `ContribWiring` (a test's router). */
export function StableNavigateProvider({ children }: { children: ReactNode }) {
  return <StableNavigateContext value={useStableNavigateSource()}>{children}</StableNavigateContext>
}
