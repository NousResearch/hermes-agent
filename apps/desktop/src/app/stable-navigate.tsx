/**
 * react-router's `useNavigate()` returns a new function on every location change (it closes over the
 * current pathname to resolve relative targets), so every component that only SENDS the user
 * somewhere re-rendered on each navigation. The shell navigates from outside `<Routes>`, where a
 * relative target resolves against the root either way, so one navigate captured where the shell
 * already follows the location serves every such caller and never changes identity.
 */
import { useLayoutEffect, useRef, useState } from 'react'
import { type NavigateFunction, type NavigateOptions, type To, useNavigate } from 'react-router'

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
