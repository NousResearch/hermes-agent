/**
 * Where a tip can actually point right now.
 *
 * A tip has no spotlight and no scrim, so a bubble anchored to something the
 * user cannot see is not a dimmed step — it is an arrow into empty space. The
 * anchor is therefore re-resolved from the DOM every time, and a tip whose
 * handles are all missing is simply skipped this round.
 *
 * The `data-pane-hidden` guard is the same one the tour collector owes: an
 * inactive tab in a keep-alive stack stays MOUNTED under `visibility: hidden`
 * so its scroll position survives, which means it keeps its layout box and its
 * rect is identical to the live tab's. No geometry test separates them.
 *
 * Nor does geometry see what is on top. A chat kept mounted under a page, a splash over the stage, a
 * sheet over the sidebar: the anchor underneath keeps its box, on screen, at full size, and a
 * bubble came up pointing at a place where something else was showing. So the anchor must not sit
 * in a region taken out of use (`inert`, `aria-hidden`), and the point at its middle must hit the
 * anchor itself, not whatever covers it.
 */

/** The first target that resolves to a visible, on-screen element. */
export function resolveTipAnchor(doc: Document, targets: readonly string[]): HTMLElement | null {
  for (const selector of targets) {
    const element = doc.querySelector<HTMLElement>(selector)

    if (element && isTipAnchorVisible(element)) {
      return element
    }
  }

  return null
}

function isTipAnchorVisible(element: HTMLElement): boolean {
  if (element.closest('[data-pane-hidden], [inert], [aria-hidden="true"]')) {
    return false
  }

  const rect = element.getBoundingClientRect()

  if (rect.width < 8 || rect.height < 8) {
    return false
  }

  const view = element.ownerDocument.defaultView

  if (!view) {
    return true
  }

  if (!(rect.bottom > 0 && rect.top < view.innerHeight && rect.right > 0 && rect.left < view.innerWidth)) {
    return false
  }

  // The middle of what is on screen of it: an anchor half off the window's edge is still hit there.
  const x = (Math.max(rect.left, 0) + Math.min(rect.right, view.innerWidth)) / 2
  const y = (Math.max(rect.top, 0) + Math.min(rect.bottom, view.innerHeight)) / 2
  const hit = element.ownerDocument.elementFromPoint(x, y)

  // A pointer-transparent anchor is hit through to what is under it: its own container still counts.
  return hit !== null && (element.contains(hit) || hit.contains(element))
}
