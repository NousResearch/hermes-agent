import { type RefObject, useEffect } from 'react'

/**
 * Whether the OS window should hand the mouse to whatever is behind it.
 *
 * `hit` is what the document reports under the cursor, `active` what holds
 * focus; both are answered against the shell by where they sit in the tree.
 *
 * - Anything that CONTAINS the shell is the scaffolding the HUD hangs in — the
 *   React mount, `<body>`, the document. Those are full-window and
 *   hit-testable, so a hit on one means the cursor is over nothing. Naming them
 *   was the earlier mistake: the list had `<body>` and `<html>` and missed
 *   `#root`, so every point in the window reported something and the window
 *   never went transparent at all.
 * - Focus BESIDE the shell is a portalled dialog, popover or menu, and it owns
 *   the next click — including the one outside itself that dismisses it, which
 *   the hit test cannot see coming. That pins the window solid.
 * - Focus INSIDE the shell does not. The composer holding the caret is the
 *   HUD's resting state rather than a claim on the whole rectangle, and reading
 *   it as one is what made an engaged HUD eat every click in its own empty
 *   space — on a fresh thread, the entire window.
 */
export function hudIgnoresMouse(
  root: Element,
  hit: Element | null,
  active: Element | null,
  windowFocused: boolean
): boolean {
  const overSomething = hit !== null && !hit.contains(root)
  // `windowFocused` is what stops a stale `active` — the composer keeps focus
  // after you click away to another app — pinning the HUD solid forever.
  const overlayFocused = windowFocused && active !== null && !root.contains(active) && !active.contains(root)

  return !overSomething && !overlayFocused
}

/**
 * Let clicks fall through the HUD everywhere it isn't really there.
 *
 * The one thing about HUD mode that CSS cannot express, because it is a
 * property of the OS WINDOW rather than of the page. It asks the document what
 * is under the cursor rather than keeping a second copy of the answer, so the
 * stylesheet stays the one place that decides what the HUD is made of.
 *
 * An always-on-top window eats every click inside its rectangle, visible or
 * not — and most of the HUD's rectangle is a faded-out band over whatever the
 * user is actually working in. `pointer-events: none` doesn't help: that is a
 * page-level property, and the click never reaches the page.
 *
 * So the window itself is made mouse-transparent except where it is genuinely
 * interactive: wherever the cursor is over something the HUD paints, and
 * whenever a portalled overlay holds focus. `forward: true` keeps mousemove
 * flowing while ignoring, which is what lets it re-arm when the cursor comes
 * back to the bar.
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
    // — the shell's dead space, the sheet, the faded band — is
    // `pointer-events: none` in the stylesheet, so whatever the document hands
    // back is something real: the bar, a control, the exit chip, a popover, a
    // dialog. Listing those instead is how links and dialogs ended up
    // unclickable, since portalled overlays live outside the shell.
    const apply = () => {
      const hit = point ? document.elementFromPoint(point.x, point.y) : null
      const next = hudIgnoresMouse(root, hit, document.activeElement, document.hasFocus())

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
