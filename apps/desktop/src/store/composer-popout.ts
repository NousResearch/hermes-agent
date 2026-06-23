import { atom } from 'nanostores'

import { persistBoolean, persistString, storedBoolean, storedString } from '@/lib/storage'

const POPOUT_ENABLED_STORAGE_KEY = 'hermes.desktop.composerPopout.enabled'
const POPOUT_POSITION_STORAGE_KEY = 'hermes.desktop.composerPopout.position'

/** Where the floating composer's bottom-right corner sits, measured as an inset
 *  from the viewport's bottom/right edges. Anchoring to the bottom-right keeps
 *  the box visually pinned to its default corner as the window resizes and as
 *  the box grows upward while typing (the corner stays put, height climbs). */
export interface PopoutPosition {
  bottom: number
  right: number
}

// Floating composer width (rem). Shared by the inline style that sets
// --composer-popout-width and the peel-off drag math.
export const POPOUT_WIDTH_REM = 19.5

// Default pop-out placement: tucked into the bottom-right of the thread, clear
// of the window chrome. Matches the brief's "default to the right bottom".
const DEFAULT_POSITION: PopoutPosition = { bottom: 24, right: 24 }

function readPosition(): PopoutPosition {
  const raw = storedString(POPOUT_POSITION_STORAGE_KEY)

  if (!raw) {
    return DEFAULT_POSITION
  }

  try {
    const parsed = JSON.parse(raw) as Partial<PopoutPosition>

    if (typeof parsed.bottom === 'number' && typeof parsed.right === 'number') {
      // Clamp on load — a position persisted on a larger/other monitor must not
      // strand the box off-screen on this one.
      return clampPosition({ bottom: parsed.bottom, right: parsed.right })
    }
  } catch {
    // Corrupt value — fall back to the default corner.
  }

  return DEFAULT_POSITION
}

export interface PopoutSize {
  height: number
  width: number
}

/** Viewport-space rect the floating composer is confined to. Defaults to the
 *  whole window; pass the thread area so the box can't slide under a pinned
 *  sidebar or behind the header. */
export interface PopoutBounds {
  bottom: number
  left: number
  right: number
  top: number
}

interface SetPositionOptions {
  /** Thread-area rect to confine the box to; falls back to the full window. */
  area?: PopoutBounds
  persist?: boolean
  /** Measured box size; falls back to the compact width + a min height so the
   *  box stays grabbable even when the caller can't measure it. */
  size?: PopoutSize
}

// Keep at least this much between the box and every edge of its bounds, so the
// floating composer can never be dragged (or restored) out of reach.
const EDGE_MARGIN = 8
// Height floor used when the real box height is unknown (init / load / peel-off).
export const POPOUT_ESTIMATED_HEIGHT = 56
const MIN_VISIBLE_HEIGHT = POPOUT_ESTIMATED_HEIGHT

const clampRange = (value: number, lo: number, hi: number) => Math.min(Math.max(value, lo), Math.max(lo, hi))

const rootFontSize = () => parseFloat(getComputedStyle(document.documentElement).fontSize) || 16

/** The thread area's viewport rect (excludes a pinned sidebar + the header), or
 *  undefined before it mounts — callers then fall back to the full window. */
export function readPopoutBounds(composer: Element | null): PopoutBounds | undefined {
  const el = (composer?.parentElement ?? document).querySelector('[data-slot="composer-bounds"]')

  if (!el) {
    return undefined
  }

  const { bottom, height, left, right, top, width } = el.getBoundingClientRect()

  // Pre-layout (mount before first layout) the rect is empty — fall back to the
  // window rather than clamping the box into a collapsed area.
  return width > 0 && height > 0 ? { bottom, left, right, top } : undefined
}

// Bound the bottom/right inset so the WHOLE box stays inside `area` (the thread
// region, or the window by default) — the corner anchor alone would let the
// box's width/height push it past the opposite edges.
function clampPosition({ bottom, right }: PopoutPosition, size?: PopoutSize, area?: PopoutBounds): PopoutPosition {
  const width = size?.width || POPOUT_WIDTH_REM * rootFontSize()
  const height = size?.height || MIN_VISIBLE_HEIGHT
  const { innerHeight: vh, innerWidth: vw } = window
  const a = area ?? { bottom: vh, left: 0, right: vw, top: 0 }

  return {
    bottom: clampRange(bottom, vh - a.bottom + EDGE_MARGIN, vh - a.top - height - EDGE_MARGIN),
    right: clampRange(right, vw - a.right + EDGE_MARGIN, vw - a.left - width - EDGE_MARGIN)
  }
}

export const $composerPoppedOut = atom(storedBoolean(POPOUT_ENABLED_STORAGE_KEY, false))
export const $composerPopoutPosition = atom<PopoutPosition>(readPosition())

export function setComposerPoppedOut(value: boolean) {
  $composerPoppedOut.set(value)
  persistBoolean(POPOUT_ENABLED_STORAGE_KEY, value)
}

/** Move the box (state only by default). Used per-frame during a drag — no IO
 *  unless `persist`. Returns the clamped position so callers can sync their live
 *  ref. Pass the measured `size` for exact bounds; otherwise a fallback keeps it
 *  on-screen. */
export function setComposerPopoutPosition(position: PopoutPosition, { area, persist, size }: SetPositionOptions = {}): PopoutPosition {
  const next = clampPosition(position, size, area)
  $composerPopoutPosition.set(next)

  if (persist) {
    persistString(POPOUT_POSITION_STORAGE_KEY, JSON.stringify(next))
  }

  return next
}
