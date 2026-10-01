import { useEffect, useState } from 'react'

/** Debounce a fast-changing value (search input, slider, …) so effects/queries
 *  keyed on it only fire once the value settles. */
export function useDebounced<T>(value: T, delayMs: number): T {
  const [debounced, setDebounced] = useState(value)

  useEffect(() => {
    const handle = setTimeout(() => setDebounced(value), delayMs)

    return () => clearTimeout(handle)
  }, [value, delayMs])

  return debounced
}
