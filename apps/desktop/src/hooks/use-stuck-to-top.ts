import { type RefObject, useEffect, useState } from 'react'

/** Nearest scrollable ancestor (the IntersectionObserver root). */
function scrollParent(el: Element | null): Element | null {
  let node = el?.parentElement ?? null

  while (node) {
    const overflowY = getComputedStyle(node).overflowY

    if (overflowY === 'auto' || overflowY === 'scroll') {
      return node
    }

    node = node.parentElement
  }

  return null
}

/**
 * True while `ref` is pinned at the top of its scroll container by
 * `position: sticky`. Detects it with a zero-height sentinel inserted just
 * above the element: once the sentinel scrolls out under the sticky offset, the
 * element is stuck. `stickyTopPx` is the element's `top` offset so the sentinel
 * trips exactly when the element parks. CSS-native — no scroll/pointer math.
 */
export function useStuckToTop(ref: RefObject<HTMLElement | null>, stickyTopPx = 0): boolean {
  const [stuck, setStuck] = useState(false)

  useEffect(() => {
    const el = ref.current

    if (!el || typeof IntersectionObserver === 'undefined') {
      return
    }

    const root = scrollParent(el)
    const sentinel = document.createElement('div')
    sentinel.setAttribute('aria-hidden', 'true')
    sentinel.style.cssText = 'position:absolute;top:0;left:0;height:1px;width:1px;pointer-events:none;'
    el.style.position ||= 'relative'
    el.prepend(sentinel)

    const observer = new IntersectionObserver(
      ([entry]) => setStuck(entry.intersectionRatio === 0),
      // Pull the root's top edge down by the sticky offset so the sentinel
      // leaves the observed band exactly when the element parks.
      { root, rootMargin: `-${stickyTopPx + 1}px 0px 0px 0px`, threshold: [0, 1] }
    )

    observer.observe(sentinel)

    return () => {
      observer.disconnect()
      sentinel.remove()
    }
  }, [ref, stickyTopPx])

  return stuck
}
