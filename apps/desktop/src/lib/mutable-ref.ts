import type { MutableRefObject } from 'react'

/** Imperative ref write — extracted so react-compiler doesn't flag hook-arg refs. */
export function setMutableRef<T>(ref: MutableRefObject<T>, value: T) {
  ref.current = value
}
