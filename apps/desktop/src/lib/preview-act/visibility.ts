/**
 * VISIBILITY — is this element actually there, and is it on screen.
 *
 * Two different questions with two different answers. `visible` decides whether
 * an element belongs in the inventory at all: painted, not screen-reader-only,
 * not buried under something else. `onScreen` decides whether the overlay
 * should draw a box around it, which is a narrower question — the inventory
 * includes things below the fold the agent will scroll to, and the overlay
 * does not.
 *
 * Most of what is here exists because of a specific class of thing that looks
 * interactive and is not: the sr-only recipe in both its spellings, a control
 * inside an `aria-hidden` subtree, a 1px box parked at the document origin, an
 * element under a cookie wall. Each one of those, admitted, is an action the
 * agent takes that visibly does nothing.
 *
 * SELF-CONTAINMENT: see `naming.ts`. Same contract, same reason for the
 * factory shape.
 */

export interface VisibilityKit {
  /** In the viewport right now, and not a page-sized wrapper. */
  onScreen(el: Element): boolean
  /** Painted at all, ancestors included. */
  shown(el: Element): boolean
  /** Painted, reachable, big enough to aim at, and not buried. */
  visible(el: Element): boolean
}

/** Build the visibility tests against one document and its view. */
export function visibilityKit(doc: Document, win: null | Window): VisibilityKit {
  /** On screen right now, and worth drawing a box around. The field is strictly
   *  viewport-bound: a mark past the fold is one nobody can see, it spends the
   *  budget that should have gone to what IS on screen, and the ones parked far
   *  off to the side were landing as stray labels in the corner. */
  const onScreen = (el: Element): boolean => {
    if (!win) {
      return false
    }

    const box = el.getBoundingClientRect()

    if (box.right <= 0 || box.bottom <= 0 || box.left >= win.innerWidth || box.top >= win.innerHeight) {
      return false
    }

    // Page-sized wrappers. Outlining one draws a rectangle around the whole
    // screen, which says nothing and frames everything inside it as if it
    // mattered. Full-width banners are fine — it takes both dimensions.
    return !(box.width >= win.innerWidth * 0.95 && box.height >= win.innerHeight * 0.9)
  }

  /** Painted at all, ancestors included. */
  const shown = (el: Element): boolean => {
    // Folds in display/visibility/opacity/content-visibility inherited from an
    // ANCESTOR, which this element's own computed style does not report: inside
    // a parent at opacity 0, every child still reads back opacity 1.
    const seen = (el as Element & { checkVisibility?: (opts: object) => boolean }).checkVisibility

    if (typeof seen === 'function' && !seen.call(el, { checkOpacity: true, checkVisibilityCSS: true })) {
      return false
    }

    const style = win && win.getComputedStyle ? win.getComputedStyle(el) : null

    if (style) {
      if (style.display === 'none' || style.visibility === 'hidden' || style.opacity === '0') {
        return false
      }

      // The screen-reader-only recipe, both spellings: the deprecated `clip` and
      // the modern `clip-path: inset(50%)`. Neither exists for any purpose other
      // than hiding something visually while keeping it focusable.
      if ((style.clip && style.clip !== 'auto') || (style.clipPath || '').indexOf('inset(50%') !== -1) {
        return false
      }
    }

    return true
  }

  const visible = (el: Element): boolean => {
    if (!shown(el)) {
      return false
    }

    // Things the page itself declares are not for interacting with, neither of
    // which shows up in a computed style or a bounding box: an aria-hidden
    // subtree is invisible to every other assistive client, and `inert` cannot
    // be clicked at all. Disabled controls are deliberately NOT filtered here —
    // they stay in the inventory so the agent is told a button is disabled
    // rather than hunting for one that appears not to exist.
    if (el.closest('[aria-hidden="true"], [inert]')) {
      return false
    }

    const rect = el.getBoundingClientRect()

    // Below the fold is fine — we scroll to it. Off to the LEFT or ABOVE is the
    // other half of the sr-only trick (`left: -9999px`, `top: -9999px`) and
    // scrolling never brings those back.
    if (rect.right <= 0 || rect.bottom <= 0) {
      return false
    }

    // Bigger than the 1px box sr-only collapses to. Nothing a person can aim at
    // is 2px across, and admitting those is what put the cursor in the corner:
    // they cluster at the document origin, so they sort to the FRONT of the
    // inventory and the agent reaches for one as the first thing on the page.
    if (rect.width < 3 || rect.height < 3) {
      return false
    }

    // Occlusion, and the last filter for a reason: everything above is cheap
    // and this one forces layout. If the middle of the element is on screen,
    // whatever the browser reports at that point has to BE this element — its
    // own descendant (an icon inside a link) or the box it paints inside are
    // fine, anything else means it is buried under a sticky header, a cookie
    // wall, or a transparent layer the page put on top. Those are exactly the
    // targets the agent aims at and then misses. Elements below the fold cannot
    // be hit-tested from here, so they are taken on trust and land back in this
    // function after the scroll.
    const midX = rect.left + rect.width / 2
    const midY = rect.top + rect.height / 2

    const under = (doc as Document & { elementFromPoint?: (x: number, y: number) => Element | null }).elementFromPoint

    if (
      typeof under === 'function' &&
      win &&
      midX >= 0 &&
      midY >= 0 &&
      midX < win.innerWidth &&
      midY < win.innerHeight
    ) {
      const over = under.call(doc, midX, midY)

      if (!over || !(over === el || el.contains(over) || over.contains(el))) {
        return false
      }
    }

    return true
  }

  return { onScreen, shown, visible }
}
