/**
 * Backdrop image opacity.
 *
 * One lever, 0-100. 100 keeps the current backdrop strength; 0 makes the
 * backdrop image transparent. This is a renderer-only display preference and
 * does not affect native window opacity.
 */

import { atom } from 'nanostores'

import { persistString, storedString } from '@/lib/storage'

const KEY = 'hermes.desktop.backdrop-opacity.v1'

const clamp = (n: number): number => Math.min(100, Math.max(0, Math.round(n)))

const read = (): number => {
  const stored = storedString(KEY)
  const n = stored === null ? Number.NaN : Number(stored)

  return Number.isFinite(n) ? clamp(n) : 100
}

export const $backdropOpacity = atom<number>(typeof window === 'undefined' ? 100 : read())

export function setBackdropOpacity(opacity: number): void {
  $backdropOpacity.set(clamp(opacity))
}

if (typeof window !== 'undefined') {
  $backdropOpacity.subscribe(opacity => {
    persistString(KEY, String(opacity))
  })
}
