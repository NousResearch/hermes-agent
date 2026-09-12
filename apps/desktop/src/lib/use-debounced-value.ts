import { useEffect, useState } from 'react'

/** Delays value propagation until it settles ~`delayMs` after the last change:
 *  the input stays instant while expensive filtering (thousands of catalog
 *  rows) runs at most once per pause instead of per keystroke. */
export function useDebouncedValue<T>(value: T, delayMs = 50): T {
  const [settled, setSettled] = useState(value)

  useEffect(() => {
    const timer = setTimeout(() => setSettled(value), delayMs)

    return () => clearTimeout(timer)
  }, [value, delayMs])

  return settled
}
