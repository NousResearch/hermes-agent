import { useCallback, useEffect, useRef, useState } from 'react'

interface Options {
  /** How long the send waits. 0 (or less) commits on the keystroke. */
  graceMs: number
  /** Commit the draft. Called when the hold expires, or immediately when there
   *  is no hold — the caller's own send path either way. */
  onCommit: () => void
}

/**
 * Holds an inferred send for a beat so it can be taken back.
 *
 * The payload deliberately stays in the composer for the whole hold: the draft
 * is the source of truth until `onCommit` runs, so cancelling is "do nothing"
 * rather than an undo — nothing has been submitted, cleared, or stashed, and
 * attachments cannot be lost because they never moved.
 *
 * `hold()` reports whether it actually held. A caller with no window to offer
 * (graceMs 0) gets an immediate commit and `false`, so a haptic can fire on the
 * wrong signal otherwise.
 */
export function useComposerSendGrace({ graceMs, onCommit }: Options) {
  const [holding, setHolding] = useState(false)
  const timerRef = useRef<number | undefined>(undefined)
  // Kept fresh without re-creating the timer callbacks mid-hold.
  const commitRef = useRef(onCommit)

  commitRef.current = onCommit

  const cancel = useCallback(() => {
    if (timerRef.current !== undefined) {
      window.clearTimeout(timerRef.current)
      timerRef.current = undefined
    }

    setHolding(false)
  }, [])

  const hold = useCallback((): boolean => {
    cancel()

    if (graceMs <= 0) {
      commitRef.current()

      return false
    }

    setHolding(true)
    timerRef.current = window.setTimeout(() => {
      timerRef.current = undefined
      setHolding(false)
      commitRef.current()
    }, graceMs)

    return true
  }, [cancel, graceMs])

  // A composer that unmounts mid-hold (session swap, pane close) must not fire
  // into a dead tree.
  useEffect(() => cancel, [cancel])

  return { cancel, hold, holding }
}
