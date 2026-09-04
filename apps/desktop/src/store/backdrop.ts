import { atom } from 'nanostores'

import { persistBoolean, readKey, storedBoolean, writeKey } from '@/lib/storage'

const KEY = 'hermes.desktop.backdrop.v1'
const OPACITY_KEY = 'hermes.desktop.backdrop-opacity.v1'

/** Whether the faint statue image renders behind the chat transcript. */
export const $backdrop = atom(storedBoolean(KEY, false))

$backdrop.subscribe(on => persistBoolean(KEY, on))

export function setBackdrop(on: boolean) {
  $backdrop.set(on)
}

// The image ships tuned for a barely-there watermark (2.5%), so that stays the
// default for anyone who never touches the new lever — only an explicit visit
// to Settings changes what's on screen.
export const BACKDROP_OPACITY_DEFAULT = 2.5
export const BACKDROP_OPACITY_MIN = 0
export const BACKDROP_OPACITY_MAX = 100
export const BACKDROP_OPACITY_STEP = 0.5

export function clampBackdropOpacity(value: number): number {
  if (!Number.isFinite(value)) {
    return BACKDROP_OPACITY_DEFAULT
  }

  return Math.min(BACKDROP_OPACITY_MAX, Math.max(BACKDROP_OPACITY_MIN, value))
}

const storedOpacity = (): number => {
  const raw = readKey(OPACITY_KEY)

  if (raw === null) {
    return BACKDROP_OPACITY_DEFAULT
  }

  const parsed = Number(raw)

  return Number.isFinite(parsed) ? clampBackdropOpacity(parsed) : BACKDROP_OPACITY_DEFAULT
}

/** How visible the statue image is, 0–100 (percent), independent of the on/off toggle. */
export const $backdropOpacity = atom(storedOpacity())

$backdropOpacity.subscribe(value => writeKey(OPACITY_KEY, String(value)))

export function setBackdropOpacity(value: number) {
  $backdropOpacity.set(clampBackdropOpacity(value))
}
