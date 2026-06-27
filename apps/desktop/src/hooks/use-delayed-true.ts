import { useEffect, useState } from 'react'

/**
 * Returns true only after `active` has stayed true continuously for `delayMs`.
 * Flips back to false the instant `active` goes false. Use it to gate loading
 * skeletons so a fast operation doesn't flash one — the UI just stays blank for
 * the (sub-perceptible) delay window, and the skeleton appears only when a load
 * is genuinely slow.
 */
export function useDelayedTrue(active: boolean, delayMs = 180): boolean {
  const [shown, setShown] = useState(false)

  useEffect(() => {
    if (!active) {
      setShown(false)

      return
    }

    const id = window.setTimeout(() => setShown(true), delayMs)

    return () => window.clearTimeout(id)
  }, [active, delayMs])

  return shown
}
