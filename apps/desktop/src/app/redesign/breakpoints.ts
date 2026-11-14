import { atom } from 'nanostores'

/** The redesign's responsive model (DESIGN.md › "Responsive"): three width classes, two breakpoints.
 *
 *  - compact (< 640): phones and narrow docked windows. No rail column; the band carries a menu
 *    button that opens the left side of the desktop layout (rail + panel) as a sheet.
 *  - medium (640–1023): the rail stays; the chat sidebar and page panels become sheets over a
 *    full-width stage.
 *  - wide (≥ 1024): the desktop layout.
 *
 *  CSS cannot read custom properties in a media query, so stylesheets repeat these two literals
 *  (`max-width: 639.98px` / `max-width: 1023.98px`) with a comment pointing here; tokens/space.css
 *  documents them as `--portal-breakpoint-medium` / `--portal-breakpoint-wide`. The classic shell's
 *  sidebar dock point (layout-constants.ts) is the medium breakpoint. */
export const BREAKPOINT_MEDIUM_PX = 640
export const BREAKPOINT_WIDE_PX = 1024

export type WidthClass = 'compact' | 'medium' | 'wide'

/** `max-width` is inclusive: shave a hair so a window exactly at a breakpoint takes the larger class. */
export const COMPACT_QUERY = `(max-width: ${BREAKPOINT_MEDIUM_PX - 0.02}px)`
export const BELOW_WIDE_QUERY = `(max-width: ${BREAKPOINT_WIDE_PX - 0.02}px)`

/** A short window (a landscape phone, a raised software keyboard): the composer goes to one row and
 *  the browser band shrinks (CSS repeats this literal as `max-height: 30rem`). */
export const SHORT_QUERY = '(max-height: 30rem)'

export function widthClassFor(width: number): WidthClass {
  if (width < BREAKPOINT_MEDIUM_PX) {
    return 'compact'
  }

  return width < BREAKPOINT_WIDE_PX ? 'medium' : 'wide'
}

// Guarded like every matchMedia call site: node and jsdom test envs may have no `matchMedia`.
const query = (media: string) => (typeof window !== 'undefined' ? window.matchMedia?.(media) : undefined)

const compactQuery = query(COMPACT_QUERY)
const belowWideQuery = query(BELOW_WIDE_QUERY)

function currentWidthClass(): WidthClass {
  if (compactQuery?.matches) {
    return 'compact'
  }

  return belowWideQuery?.matches ? 'medium' : 'wide'
}

/** This window's width class, live. Two media-query listeners, not a resize listener: it changes
 *  only when a breakpoint is crossed. */
export const $widthClass = atom<WidthClass>(currentWidthClass())

const sync = () => $widthClass.set(currentWidthClass())

compactQuery?.addEventListener('change', sync)
belowWideQuery?.addEventListener('change', sync)

const shortQuery = query(SHORT_QUERY)

/** Whether this window is short, live. */
export const $shortViewport = atom(Boolean(shortQuery?.matches))

shortQuery?.addEventListener('change', event => $shortViewport.set(event.matches))
