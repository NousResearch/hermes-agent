import { type RefObject, useEffect } from 'react'

/**
 * Let clicks fall through the HUD everywhere it isn't really there.
 *
 * The one thing about HUD mode that CSS cannot express, because it is a
 * property of the OS WINDOW rather than of the page. It reads the engaged state
 * off the DOM (`:focus-within`) rather than keeping a second copy, so there is
 * one answer to "is the HUD in use" and the stylesheet owns it.
 *
 * An always-on-top window eats every click inside its rectangle, visible or
 * not — and most of the HUD's rectangle is a faded-out band over whatever the
 * user is actually working in. `pointer-events: none` doesn't help: that is a
 * page-level property, and the click never reaches the page.
 *
 * So the window itself is made mouse-transparent except where it is genuinely
 * interactive: the bar, always, and everything else only while the composer
 * holds focus — the same line the band and its exit chip draw with
 * `pointer-events`. `forward: true` keeps mousemove flowing while ignoring,
 * which is what lets it re-arm when the cursor comes back to the bar.
 */
export function useHudClickThrough(rootRef: RefObject<HTMLElement | null>): void {
  useEffect(() => {
    const root = rootRef.current
    const setIgnoreMouse = window.hermesDesktop?.hud?.setIgnoreMouse

    if (!root || !setIgnoreMouse) {
      return
    }

    let ignoring: boolean | null = null
    // Where the cursor was last seen, so a focus change can re-decide without
    // waiting for the next move (blurring with the cursor parked on the bar
    // must not make the bar untouchable until you jiggle the mouse).
    let point: { x: number; y: number } | null = null

    const overBar = () => {
      const bar = root.querySelector('[data-slot="composer-dock"]')

      if (!bar || !point) {
        return false
      }

      const rect = bar.getBoundingClientRect()

      return point.x >= rect.left && point.x <= rect.right && point.y >= rect.top && point.y <= rect.bottom
    }

    const apply = () => {
      const next = !root.matches(':focus-within') && !overBar()

      if (ignoring !== next) {
        ignoring = next
        setIgnoreMouse(next)
      }
    }

    const onMove = (event: MouseEvent) => {
      point = { x: event.clientX, y: event.clientY }
      apply()
    }

    apply()
    window.addEventListener('mousemove', onMove)
    root.addEventListener('focusin', apply)
    root.addEventListener('focusout', apply)

    return () => {
      setIgnoreMouse(false)
      window.removeEventListener('mousemove', onMove)
      root.removeEventListener('focusin', apply)
      root.removeEventListener('focusout', apply)
    }
  }, [rootRef])
}
