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
// --composer-popout-width and the peel-off drag math (to center it on the cursor).
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

interface SetPositionOptions {
  persist?: boolean
  /** Measured box size; falls back to the compact width + a min height so the
   *  box stays grabbable even when the caller can't measure it. */
  size?: PopoutSize
}

// Keep at least this much of every edge between the box and the viewport, so the
// floating composer can never be dragged (or restored) out of reach.
const EDGE_MARGIN = 8
// Height floor used when the real box height is unknown (init / load).
const MIN_VISIBLE_HEIGHT = 56

const clampRange = (value: number, lo: number, hi: number) => Math.min(Math.max(value, lo), Math.max(lo, hi))

const rootFontSize = () => parseFloat(getComputedStyle(document.documentElement).fontSize) || 16

// Bound the bottom-right inset so the WHOLE box stays on-screen — the corner
// anchor alone would let the box's width/height push it past the left/top edges.
function clampPosition({ bottom, right }: PopoutPosition, size?: PopoutSize): PopoutPosition {
  const width = size?.width || POPOUT_WIDTH_REM * rootFontSize()
  const height = size?.height || MIN_VISIBLE_HEIGHT

  return {
    bottom: clampRange(bottom, EDGE_MARGIN, window.innerHeight - height - EDGE_MARGIN),
    right: clampRange(right, EDGE_MARGIN, window.innerWidth - width - EDGE_MARGIN)
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
export function setComposerPopoutPosition(position: PopoutPosition, { persist, size }: SetPositionOptions = {}): PopoutPosition {
  const next = clampPosition(position, size)
  $composerPopoutPosition.set(next)

  if (persist) {
    persistString(POPOUT_POSITION_STORAGE_KEY, JSON.stringify(next))
  }

  return next
}
