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
      return { bottom: parsed.bottom, right: parsed.right }
    }
  } catch {
    // Corrupt value — fall back to the default corner.
  }

  return DEFAULT_POSITION
}

export const $composerPoppedOut = atom(storedBoolean(POPOUT_ENABLED_STORAGE_KEY, false))
export const $composerPopoutPosition = atom<PopoutPosition>(readPosition())

export function setComposerPoppedOut(value: boolean) {
  $composerPoppedOut.set(value)
  persistBoolean(POPOUT_ENABLED_STORAGE_KEY, value)
}

const clamp = (value: number, max: number) => Math.min(Math.max(0, value), Math.max(0, max))

// Clamp the corner inset so a viewport shrink (or a stale persisted value) can't
// strand the box fully off-screen.
const clampPosition = ({ bottom, right }: PopoutPosition): PopoutPosition => ({
  bottom: clamp(bottom, window.innerHeight - 60),
  right: clamp(right, window.innerWidth - 80)
})

/** Move the box (state only). Used per-frame during a drag — no IO. Returns the
 *  clamped position so callers can keep their live ref in sync. */
export function setComposerPopoutPosition(position: PopoutPosition, persist = false): PopoutPosition {
  const next = clampPosition(position)
  $composerPopoutPosition.set(next)

  if (persist) {
    persistString(POPOUT_POSITION_STORAGE_KEY, JSON.stringify(next))
  }

  return next
}
