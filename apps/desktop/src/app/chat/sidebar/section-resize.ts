// Drag-resizing the Sessions section of the chat sidebar. Pure geometry, no
// DOM: the sash measures, these functions decide, the sash applies.
//
// Sessions sits between two neighbours — Pinned above, the bottom block
// (messaging platforms + cron, one unit) below — and each seam is a true
// splitter: the neighbour gives up exactly what Sessions gains, so the seam
// tracks the pointer and Sessions' far edge stays put.

/** Pane-store keys. Presentation-only, so they live with the other pane sizes
 *  (per interface mode, cleared by a layout reset). */
export const SIDEBAR_PINNED_SECTION_ID = 'sidebar-section:pinned'
export const SIDEBAR_SESSIONS_SECTION_ID = 'sidebar-section:sessions'
export const SIDEBAR_BOTTOM_SECTION_ID = 'sidebar-section:bottom'

/** Mirrors the Sessions root's `min-h-32` floor. */
export const SESSIONS_MIN_PX = 128
/** Roughly one row or header — a seam never hides a neighbour entirely. */
export const NEIGHBOUR_MIN_PX = 32
export const KEYBOARD_STEP_PX = 16

export interface SeamStart {
  /** The neighbour's visible height when the drag began. */
  neighbour: number
  /** Sessions section's height when the drag began. */
  sessions: number
}

export interface SeamHeights {
  neighbour: number
  sessions: number
}

/**
 * Grow the neighbour by `delta` px (negative shrinks it) and hand Sessions the
 * difference. Pinned's seam passes the pointer's dy; the bottom block's seam
 * passes -dy, since dragging it down shrinks the block.
 *
 * The seam moves freely both ways until one SIDE hits its floor — the
 * neighbour's one row, or Sessions' own minimum. A neighbour may grow past its
 * content (the slack shows as room below its rows), as any splitter pane does.
 */
export function resizeSessionsSeam(start: SeamStart, delta: number): SeamHeights {
  const ceiling = start.neighbour + Math.max(0, start.sessions - SESSIONS_MIN_PX)
  const neighbour = Math.min(ceiling, Math.max(NEIGHBOUR_MIN_PX, start.neighbour + delta))
  // Sessions trades exactly what the neighbour moved after clamping, so the
  // seam stops with whichever side hit its floor. Both stay unrounded:
  // measured heights are fractional (zoom, card rows), and rounding either
  // side breaks the trade — up overflows the list into a 1px scroll, down
  // leaks a pixel per drag, and a seam pinned against a floor creeps.
  const sessions = Math.max(SESSIONS_MIN_PX, start.sessions - (neighbour - start.neighbour))

  return { neighbour, sessions }
}
