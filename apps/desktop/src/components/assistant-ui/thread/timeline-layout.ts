import type { CSSProperties } from 'react'

const LINE_PX = 16

/**
 * The rail floats over the thread as the viewport's sibling. It stands left of
 * the viewport's scrollbar (`--timeline-gutter`, kept by `syncRailGutter`) and
 * is no wider than the transcript's inline padding (`timeline.css`), so it
 * never covers the scrollbar thumb or the first glyph of a right-aligned line.
 */
export const timelineRailStyle = (count: number): CSSProperties => ({
  position: 'absolute',
  right: 'var(--timeline-gutter, 0px)',
  top: '50%',
  transform: 'translateY(-50%)',
  zIndex: 40,
  pointerEvents: 'auto',
  height: `min(${count * 0.4375}rem, 50%)`
})

/** The scrollbar comes and goes with overflow; written as a var so React never repaints for it. */
export function syncRailGutter(rail: HTMLElement, viewport: HTMLElement): void {
  const gutter = `${Math.max(0, viewport.offsetWidth - viewport.clientWidth)}px`

  if (rail.style.getPropertyValue('--timeline-gutter') !== gutter) {
    rail.style.setProperty('--timeline-gutter', gutter)
  }
}

const canScroll = (element: HTMLElement, deltaY: number): boolean =>
  deltaY < 0 ? element.scrollTop > 0 : element.scrollTop + element.clientHeight < element.scrollHeight - 1

/**
 * Being a sibling, the rail never chains a wheel to the transcript. Whatever
 * the ticks cannot take themselves (all of it on a rail that shows every tick,
 * the overflow past either end of a long one) goes to the viewport: its own
 * listeners (follow escape, show earlier, jump cancel) see it first, as they
 * would a native wheel, then the viewport moves. The copy does not bubble, so
 * bubble-phase ancestors still see one wheel. ctrl+wheel stays a zoom gesture.
 */
export function forwardRailWheel(event: WheelEvent, ticks: HTMLElement, viewport: HTMLElement): boolean {
  if (event.ctrlKey || !event.deltaY || canScroll(ticks, event.deltaY)) {
    return false
  }

  const forwarded = new WheelEvent('wheel', {
    altKey: event.altKey,
    cancelable: true,
    clientX: event.clientX,
    clientY: event.clientY,
    deltaMode: event.deltaMode,
    deltaY: event.deltaY,
    metaKey: event.metaKey,
    screenX: event.screenX,
    screenY: event.screenY,
    shiftKey: event.shiftKey
  })

  if (!viewport.dispatchEvent(forwarded)) {
    return false
  }

  const unit =
    event.deltaMode === WheelEvent.DOM_DELTA_LINE
      ? LINE_PX
      : event.deltaMode === WheelEvent.DOM_DELTA_PAGE
        ? viewport.clientHeight
        : 1

  viewport.scrollBy({ top: event.deltaY * unit })

  return true
}
