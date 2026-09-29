import { atom } from 'nanostores'

import type { Locale } from '@/i18n'
import { persistString, storedString } from '@/lib/storage'

import { clampOrbScale, orbWindowSize } from './desktop-orb-geometry'
import type { JarvisTaskPhase, JarvisVoiceState } from './types'

export interface DesktopOrbState {
  active: boolean
  connected: boolean
  locale: Locale
  voice: JarvisVoiceState
  task: JarvisTaskPhase
}

const MODE_KEY = 'hermes.desktop.pet-overlay-mode.v1'
export const $desktopOrbMode = atom(storedString(MODE_KEY) === 'orb')
$desktopOrbMode.subscribe(orb => persistString(MODE_KEY, orb ? 'orb' : 'pet'))
export const $desktopOrbConnection = atom({ active: false, connected: false, locale: 'pl' as Locale })

const SCALE_KEY = 'czesiek.orb-scale.v1'

/**
 * The orb's size is a user preference shared by two windows (the overlay changes
 * it, the main window reads it when it opens the overlay), so it is read from
 * storage on demand instead of being cached in one window's atom.
 */
export function readOrbScale(): number {
  return clampOrbScale(Number(storedString(SCALE_KEY) ?? 1))
}

export function writeOrbScale(scale: number): number {
  const next = clampOrbScale(scale)
  persistString(SCALE_KEY, String(next))

  return next
}

/** Overlay window size for the orb at the saved scale. */
export function currentOrbWindowSize() {
  return orbWindowSize(readOrbScale())
}

/** Preserve the grabbed point; never snap the orb's centre to the pointer. */
export function moveOrb(
  origin: { x: number; y: number },
  start: { x: number; y: number },
  pointer: { x: number; y: number }
) {
  return { x: Math.round(origin.x + pointer.x - start.x), y: Math.round(origin.y + pointer.y - start.y) }
}
