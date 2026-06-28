import { atom } from 'nanostores'

import { persistString, storedString } from '@/lib/storage'

const KEY = 'hermes.desktop.decorative-backdrop.v1'

export type DecorativeBackdropMode = 'off' | 'subtle' | 'full'

export const BACKDROP_OPACITY: Record<DecorativeBackdropMode, number> = {
  off: 0,
  subtle: 0.015,
  full: 0.025
}

const isDecorativeBackdropMode = (value: string | null): value is DecorativeBackdropMode =>
  value === 'off' || value === 'subtle' || value === 'full'

const read = (): DecorativeBackdropMode => {
  const value = storedString(KEY)

  return isDecorativeBackdropMode(value) ? value : 'full'
}

export const $decorativeBackdrop = atom<DecorativeBackdropMode>(typeof window === 'undefined' ? 'full' : read())

export function setDecorativeBackdrop(mode: DecorativeBackdropMode): void {
  $decorativeBackdrop.set(isDecorativeBackdropMode(mode) ? mode : 'full')
}

if (typeof window !== 'undefined') {
  $decorativeBackdrop.subscribe(mode => {
    persistString(KEY, mode)
  })
}
