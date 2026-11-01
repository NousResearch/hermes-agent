import { type RefObject, useEffect, useState } from 'react'

/**
 * Latches true the first time `ref`'s element intersects the viewport (through
 * every clipping scroll ancestor), and stays true. Expensive work that only
 * matters once seen (syntax highlighting) waits on it, so mounting a long
 * transcript does not pay for content the reader never scrolls to.
 * Without IntersectionObserver it reports true at once (nothing to wait on).
 */
export function useSeenOnce(ref: RefObject<Element | null>, enabled = true): boolean {
  const [seen, setSeen] = useState(() => typeof IntersectionObserver !== 'function')

  useEffect(() => {
    const node = ref.current

    if (!enabled || seen || !node) {
      return
    }

    const observer = new IntersectionObserver(entries => {
      if (entries.some(entry => entry.isIntersecting)) {
        observer.disconnect()
        setSeen(true)
      }
    })

    observer.observe(node)

    return () => observer.disconnect()
  }, [enabled, ref, seen])

  return seen
}
