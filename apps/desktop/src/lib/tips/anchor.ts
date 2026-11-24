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
 *
 * Covered is not gone. A menu, a dialog or a toast passes over an anchor for a moment; a pane that
 * closes takes it away. A tip that is already up waits out the first and ends on the second
 * (`hasTipAnchor`), so opening a menu does not spend a tip the user was about to read.
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

/** Whether a tip's subject is still there: one of its targets laid out in a pane that is shown,
 *  whatever stands over it for now. */
export function hasTipAnchor(doc: Document, targets: readonly string[]): boolean {
  return targets.some(selector => {
    const element = doc.querySelector<HTMLElement>(selector)

    return element !== null && isTipAnchorPresent(element)
  })
}

function isTipAnchorPresent(element: HTMLElement): boolean {
  if (element.closest('[data-pane-hidden]')) {
    return false
  }

  const rect = element.getBoundingClientRect()

  return rect.width >= 8 && rect.height >= 8
}

function isTipAnchorVisible(element: HTMLElement): boolean {
  if (!isTipAnchorPresent(element) || element.closest('[inert], [aria-hidden="true"]')) {
    return false
  }

  const rect = element.getBoundingClientRect()
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
