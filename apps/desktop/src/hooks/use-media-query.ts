import { useEffect, useState } from 'react'

export const matchesQuery = (query: string) =>
  typeof window !== 'undefined' && !!window.matchMedia && window.matchMedia(query).matches

/** Read at call time, not render time: animations check this as they start, and
 *  the OS setting can flip mid-session. */
export const prefersReducedMotion = () => matchesQuery('(prefers-reduced-motion: reduce)')

export function useMediaQuery(query: string): boolean {
  const [matches, setMatches] = useState(() => matchesQuery(query))

  useEffect(() => {
    if (typeof window === 'undefined' || !window.matchMedia) {
      return
    }

    const mql = window.matchMedia(query)
    const onChange = () => setMatches(mql.matches)

    setMatches(mql.matches)
    mql.addEventListener('change', onChange)

    return () => mql.removeEventListener('change', onChange)
  }, [query])

  return matches
}
