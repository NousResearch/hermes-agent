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
 * interactive: wherever the cursor is over something, and whenever anything in
 * the window holds focus. `forward: true` keeps mousemove flowing while
 * ignoring, which is what lets it re-arm when the cursor comes back to the bar.
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

    // Hit-test rather than enumerate. Everything the HUD doesn't want to catch
    // — the shell's dead space, the sheet, the faded band — is already
    // `pointer-events: none`, so anything the document hands back at this point
    // is something real: the bar, a control, the exit chip, a popover, a dialog.
    // Listing those instead is how links and dialogs ended up unclickable, since
    // portalled overlays live outside the shell and moving focus into one takes
    // `:focus-within` with it.
    const overSomething = () => {
      if (!point) {
        return false
      }

      const hit = document.elementFromPoint(point.x, point.y)

      return Boolean(hit) && hit !== root && hit !== document.body && hit !== document.documentElement
    }

    // Focus is asked of the document, not of the shell. A dialog or popover is
    // portalled to `document.body`, so focus entering one leaves the shell's
    // `:focus-within` — and a HUD that decides it is unused the moment it opens
    // a dialog goes mouse-transparent underneath it. Nothing but the HUD lives
    // in this window, so any focus at all is the HUD in use. `hasFocus` gates it
    // so a stale `activeElement` — the composer keeps it after you click away to
    // another app — can't pin the HUD solid forever.
    const focused = () => document.hasFocus() && document.activeElement !== document.body

    const apply = () => {
      const next = !focused() && !overSomething()

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
    document.addEventListener('focusin', apply)
    document.addEventListener('focusout', apply)

    return () => {
      setIgnoreMouse(false)
      window.removeEventListener('mousemove', onMove)
      document.removeEventListener('focusin', apply)
      document.removeEventListener('focusout', apply)
    }
  }, [rootRef])
}
