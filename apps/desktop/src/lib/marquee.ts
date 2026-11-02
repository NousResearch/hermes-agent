import type { PointerEvent } from 'react'

/** Pixels per second for the hover-marquee travel segment. */
const MARQUEE_PX_PER_SECOND = 80

/** Hover marquee: measure the actual overflow on pointerenter and arm the CSS
 *  animation (`.hover-marquee` in styles.css) only when there is some — CSS
 *  can't detect overflow on its own, and animating a non-overflowing label
 *  would wiggle for nothing. Distance-proportional duration keeps the scroll
 *  speed constant across short and long overflows. State lives in DOM
 *  attributes, not React state, so hover never re-renders the row. */
export function armMarquee(event: PointerEvent<HTMLElement>) {
  const el = event.currentTarget
  const distance = el.scrollWidth - el.clientWidth

  if (distance > 2) {
    // The keyframes spend 65% of the cycle travelling (10%→75%); scale the
    // duration so the travel segment itself moves at the target speed.
    el.style.setProperty('--marquee-d', `${distance}px`)
    el.style.setProperty('--marquee-t', `${Math.max(1, distance / MARQUEE_PX_PER_SECOND / 0.65)}s`)
    el.dataset.marquee = 'true'
  }
}

export function disarmMarquee(event: PointerEvent<HTMLElement>) {
  delete event.currentTarget.dataset.marquee
}
