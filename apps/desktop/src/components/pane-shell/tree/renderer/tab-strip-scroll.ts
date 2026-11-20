/**
 * Keeping the active tab (and the trailing "+") inside a scrolling tab strip.
 *
 * Once a zone has more tabs than fit, opening a new one appended it past the
 * right edge: the tab you just created — and the "+" that created it — were
 * both off-screen, so a second new tab meant scrolling back by hand first.
 * Activating a tab from a keybind or the session list had the same problem in
 * the other direction.
 */

import { type RefObject, useLayoutEffect } from 'react'

export interface TabStripGeometry {
  /** Visible width of the strip. */
  clientWidth: number
  /** Active tab is the LAST one, so reveal the trailing "+" along with it. */
  last: boolean
  scrollLeft: number
  /** Full scroll content: every tab plus the trailing "+". */
  scrollWidth: number
  /** Active tab's edges, measured from the start of the scroll content. */
  tabEnd: number
  tabStart: number
}

/** Where the strip should be scrolled to for the active tab to be in view —
 *  the current offset when it already is (the caller skips the write). */
export function tabStripScrollLeft({
  clientWidth,
  last,
  scrollLeft,
  scrollWidth,
  tabEnd,
  tabStart
}: TabStripGeometry): number {
  const max = Math.max(0, scrollWidth - clientWidth)

  // The last tab scrolls to the very end rather than to its own edge: the "+"
  // lives after it in the same scroll content and has to come along.
  if (last) {
    return max
  }

  if (tabStart < scrollLeft) {
    return Math.min(tabStart, max)
  }

  if (tabEnd > scrollLeft + clientWidth) {
    return Math.min(tabEnd - clientWidth, max)
  }

  return Math.min(scrollLeft, max)
}

/** Scroll `activeId`'s tab into view whenever the activation or the tab set
 *  changes. `last` drives the "+"-follows-the-final-tab case. */
export function useActiveTabVisible(
  scrollerRef: RefObject<HTMLDivElement | null>,
  activeId: string,
  { enabled, last, tabCount }: { enabled: boolean; last: boolean; tabCount: number }
): void {
  // Layout effect: the scroll write lands in the same frame as the tab's
  // insertion, so a new tab never paints off-screen first.
  useLayoutEffect(() => {
    const scroller = scrollerRef.current

    if (!enabled || !scroller) {
      return
    }

    const tab = scroller.querySelector<HTMLElement>(`[data-tree-tab="${CSS.escape(activeId)}"]`)

    if (!tab) {
      return
    }

    // Rects, not offsetLeft: the tab's offsetParent is the header strip (the
    // scroller itself isn't positioned), so offsets would be measured against
    // the wrong origin.
    const view = scroller.getBoundingClientRect()
    const rect = tab.getBoundingClientRect()
    const tabStart = rect.left - view.left + scroller.scrollLeft

    const next = tabStripScrollLeft({
      clientWidth: scroller.clientWidth,
      last,
      scrollLeft: scroller.scrollLeft,
      scrollWidth: scroller.scrollWidth,
      tabEnd: tabStart + rect.width,
      tabStart
    })

    if (next !== scroller.scrollLeft) {
      scroller.scrollLeft = next
    }
  }, [activeId, enabled, last, scrollerRef, tabCount])
}
