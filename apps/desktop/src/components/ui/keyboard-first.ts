import { useEffect, useState } from 'react'

/**
 * KEYBOARD-FIRST OVERLAYS — the shared contract for anything you open with a
 * hotkey and drive from a search field: the composer model menu, the ⌘K
 * palette, every cmdk picker.
 *
 * Both halves exist because a mouse that is merely PRESENT is not a mouse the
 * user is using.
 */

const RELEASE_EVENT = 'hermes:release-typing-focus'

/**
 * True while the pointer must be treated as absent.
 *
 * A list that opens under a parked cursor, or that re-flows under one as the
 * query filters it, fires pointerenter on whatever row happens to slide beneath
 * — and hover-selecting menus (Radix) or hover-highlighting lists take that as
 * intent, stealing the row the user typed toward. Nothing about that came from
 * the user.
 *
 * So the pointer stays inert until it actually MOVES (or scrolls, which is also
 * a hand on the mouse). One real movement hands hover back for the rest of the
 * overlay's life. Spread the result as `pointer-events-none` on the list — the
 * blunt instrument is the right one here, because it suppresses the synthetic
 * enter events at the source rather than racing them.
 *
 * Mount-scoped: overlays mount when they open, so "quiet until moved" needs no
 * knowledge of whether a hotkey or a click opened it.
 */
export function usePointerQuiet(): boolean {
  const [quiet, setQuiet] = useState(true)

  useEffect(() => {
    const wake = () => setQuiet(false)
    const options = { capture: true } as const

    window.addEventListener('mousemove', wake, options)
    window.addEventListener('wheel', wake, options)

    return () => {
      window.removeEventListener('mousemove', wake, options)
      window.removeEventListener('wheel', wake, options)
    }
  }, [])

  return quiet
}

/**
 * Hand the keyboard back to wherever the user was typing.
 *
 * Dismissing a keyboard-driven overlay ends its claim on focus. Radix returns
 * focus to the TRIGGER on close, which for the model menu is a toolbar button —
 * so committing with Enter left the next keystroke going nowhere instead of
 * into the message you were about to write. Call this on close; the composer
 * bus (subscribed once in `useKeybinds`) does the focusing, so this primitive
 * stays ignorant of what "typing" means on any given surface.
 */
export function releaseTypingFocus(): void {
  if (typeof window !== 'undefined') {
    window.dispatchEvent(new CustomEvent(RELEASE_EVENT))
  }
}

export function onReleaseTypingFocus(handler: () => void): () => void {
  if (typeof window === 'undefined') {
    return () => undefined
  }

  window.addEventListener(RELEASE_EVENT, handler)

  return () => window.removeEventListener(RELEASE_EVENT, handler)
}
