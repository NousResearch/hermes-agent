import { useStore } from '@nanostores/react'
import { useEffect, useState } from 'react'

import { $layoutScale, layoutQuery } from '@/lib/layout-scale'

export const matchesQuery = (query: string) =>
  typeof window !== 'undefined' && !!window.matchMedia && window.matchMedia(layoutQuery(query)).matches

/** Read at call time, not render time: animations check this as they start, and
 *  the OS setting can flip mid-session. */
export const prefersReducedMotion = () => matchesQuery('(prefers-reduced-motion: reduce)')

export function useMediaQuery(query: string): boolean {
  const [matches, setMatches] = useState(() => matchesQuery(query))
  // A size query is asked against the room left at the app's own scale (lib/layout-scale.ts).
  const scale = useStore($layoutScale)

  useEffect(() => {
    if (typeof window === 'undefined' || !window.matchMedia) {
      return
    }

    const mql = window.matchMedia(layoutQuery(query))
    const onChange = () => setMatches(mql.matches)

    setMatches(mql.matches)
    mql.addEventListener('change', onChange)

    return () => mql.removeEventListener('change', onChange)
  }, [query, scale])

  return matches
}
