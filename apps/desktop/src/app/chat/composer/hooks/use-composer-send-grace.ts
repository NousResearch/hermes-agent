import { useCallback, useEffect, useRef, useState } from 'react'

interface Options {
  /** How long the send waits. 0 (or less) commits on the keystroke. */
  graceMs: number
  /** Commit the draft. Called when the hold expires, or immediately when there
   *  is no hold — the caller's own send path either way. */
  onCommit: () => void
  /** Identifies the session this window belongs to. When it changes the
   *  composer has swapped sessions, and a window still waiting would resolve
   *  `onCommit` against the NEW session — so the swap cancels it. */
  ownerKey: string | null
}

/**
 * Holds an inferred send for a beat so it can be taken back.
 *
 * The payload deliberately stays in the composer for the whole hold: the draft
 * is the source of truth until `onCommit` runs, so cancelling is "do nothing"
 * rather than an undo — nothing has been submitted, cleared, or stashed, and
 * attachments cannot be lost because they never moved.
 *
 * `hold()` reports whether it took the commit — `true` whenever `onCommit` has
 * run or is about to, including the zero-window case where it runs on the spot.
 * Callers read that answer to decide whether to submit themselves, so a `false`
 * here would make them send a draft that has already gone.
 */
export function useComposerSendGrace({ graceMs, onCommit, ownerKey }: Options) {
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

    // Nothing to wait for: commit on the spot, but still report that this call
    // owns the send. A caller reading `false` as "you did not commit" would
    // submit the same draft a second time, or drain/steer the composer it just
    // emptied.
    if (graceMs <= 0) {
      commitRef.current()

      return true
    }

    setHolding(true)
    timerRef.current = window.setTimeout(() => {
      timerRef.current = undefined
      setHolding(false)
      commitRef.current()
    }, graceMs)

    return true
  }, [cancel, graceMs])

  // A composer that unmounts mid-hold (pane close) must not fire into a dead
  // tree — and one that swaps to another session while a window is still open
  // must not fire either: `commitRef` now points at the new session's send, so
  // a timer armed here would submit the wrong draft. Keying the cleanup on the
  // owner cancels the window at that handoff, not only at unmount.
  useEffect(() => cancel, [cancel, ownerKey])

  return { cancel, hold, holding }
}
