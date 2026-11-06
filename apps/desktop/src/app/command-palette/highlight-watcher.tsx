/**
 * Reports the cmdk highlight to the palette.
 *
 * cmdk calls the root `onValueChange` only in controlled mode (when the
 * `value` prop is set). This palette is uncontrolled, so that prop never
 * fires. This watcher subscribes to the cmdk store with `useCommandState`,
 * which works in both modes, and reports each highlight change.
 */

import { useCommandState } from 'cmdk'
import { useEffect, useRef } from 'react'

interface HighlightWatcherProps {
  /** Receives the cmdk value of each newly highlighted row. */
  onValue: (value: string) => void
}

export function HighlightWatcher({ onValue }: HighlightWatcherProps): null {
  const value = useCommandState(state => state.value)

  // Pin the callback so the effect fires on highlight changes only. The
  // parent passes a fresh closure on every render.
  const onValueRef = useRef(onValue)
  onValueRef.current = onValue

  useEffect(() => {
    onValueRef.current(value)
  }, [value])

  return null
}
