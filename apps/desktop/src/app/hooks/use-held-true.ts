import { useEffect, useState } from 'react'

/**
 * Immediate on true, delayed on false. Short false blips never reach the UI.
 * `resetKey` drops the hold when identity changes (session switch).
 */
export function useHeldTrue(value: boolean, holdMs: number, resetKey?: null | string): boolean {
  const [held, setHeld] = useState(value)
  const [prevKey, setPrevKey] = useState(resetKey)

  if (resetKey !== prevKey) {
    setPrevKey(resetKey)
    setHeld(value)
  }

  useEffect(() => {
    if (value) {
      setHeld(true)

      return
    }

    const timer = setTimeout(() => setHeld(false), holdMs)

    return () => clearTimeout(timer)
  }, [holdMs, resetKey, value])

  return value || held
}
