/**
 * Background image / texture overlay in the chat surface.
 *
 * The default "statue" backdrop (filler-bg0.jpg with difference blend + dither
 * noise) is a purely decorative effect.  Some users find the faint texture
 * distracting or reminiscent of a broken display, especially in dark mode.
 * This toggle lets them switch to a plain solid background.
 *
 * Persisted to localStorage so the preference survives restarts.
 */

import { atom } from 'nanostores'

import { persistString, storedString } from '@/lib/storage'

const KEY = 'hermes.desktop.backdrop.v1'

const read = (): boolean => {
  const v = storedString(KEY)
  // Default: on (preserves existing behaviour).
  if (v === null || v === '') return true
  return v !== '0'
}

export const $backdropEnabled = atom<boolean>(
  typeof window === 'undefined' ? true : read()
)

export function setBackdropEnabled(enabled: boolean): void {
  $backdropEnabled.set(enabled)
}

if (typeof window !== 'undefined') {
  $backdropEnabled.subscribe(enabled => {
    persistString(KEY, enabled ? '1' : '0')
  })
}
