import { type RefObject, useEffect } from 'react'

/** How long after the last wheel-driven scroll the transcript takes the pointer again. */
export const WHEEL_SCROLL_HOVER_REST_MS = 150

/** A scroll this soon after a wheel tick is that tick's, not a streaming reply's follow. */
const WHEEL_TO_SCROLL_MS = 100

interface WheelScrollHoverRestOptions {
  contentRef: RefObject<HTMLElement | null>
  scrollRef: RefObject<HTMLElement | null>
  paneVisible: boolean
}

/**
 * While the wheel scrolls the transcript, the pointer stands still and the turns move under it, so
 * the browser re-targets hover every frame: enter and leave events through every handler on the
 * way, and a restyle of each turn the pointer crosses (a `group-hover` rule restyles the group's
 * whole subtree). On a long chat that was about 60% of the script and 70% of the restyled elements
 * of a reading-pace scroll, all to light up affordances nobody can aim at mid-scroll.
 *
 * So the content sits the gesture out: `data-wheel-scrolling` marks it while a wheel-driven scroll
 * is under way and a skin stops it taking the pointer for that long (thread/redesign.css). It comes back when
 * the wheel rests, and at once when the pointer really moves or a button goes down, so nothing the
 * user reaches for is ever late.
 *
 * Only a scroll of THIS viewport counts. A wheel over a code block or a capped tool output scrolls
 * that box, the transcript does not move, and the box must keep the wheel. A reply streaming in
 * scrolls the transcript with no wheel at all, and the turns under it have to stay clickable.
 */
export function useWheelScrollHoverRest({ contentRef, scrollRef, paneVisible }: WheelScrollHoverRestOptions) {
  useEffect(() => {
    const viewport = scrollRef.current
    const content = contentRef.current

    if (!paneVisible || !viewport || !content) {
      return
    }

    let wheeledAt = -Infinity
    let timer = 0
    let pointer: { x: number; y: number } | null = null

    const wake = () => {
      window.clearTimeout(timer)
      timer = 0
      content.removeAttribute('data-wheel-scrolling')
    }

    const onWheel = (event: WheelEvent) => {
      // A held button is a drag (a text selection being extended): leave its hit-testing alone.
      wheeledAt = event.buttons ? -Infinity : performance.now()
    }

    const onScroll = () => {
      if (performance.now() - wheeledAt > WHEEL_TO_SCROLL_MS) {
        return
      }

      content.setAttribute('data-wheel-scrolling', '')
      window.clearTimeout(timer)
      timer = window.setTimeout(wake, WHEEL_SCROLL_HOVER_REST_MS)
    }

    // The browser re-sends the pointer's position as content scrolls under it; only a position that
    // changed is the user moving.
    const onPointerMove = (event: PointerEvent) => {
      const moved = pointer !== null && (pointer.x !== event.clientX || pointer.y !== event.clientY)
      pointer = { x: event.clientX, y: event.clientY }

      if (moved && timer) {
        wheeledAt = -Infinity
        wake()
      }
    }

    const onPointerDown = () => {
      wheeledAt = -Infinity
      wake()
    }

    viewport.addEventListener('wheel', onWheel, { passive: true })
    viewport.addEventListener('scroll', onScroll, { passive: true })
    viewport.addEventListener('pointermove', onPointerMove, { passive: true })
    viewport.addEventListener('pointerdown', onPointerDown, { passive: true })

    return () => {
      wake()
      viewport.removeEventListener('wheel', onWheel)
      viewport.removeEventListener('scroll', onScroll)
      viewport.removeEventListener('pointermove', onPointerMove)
      viewport.removeEventListener('pointerdown', onPointerDown)
    }
  }, [contentRef, paneVisible, scrollRef])
}
