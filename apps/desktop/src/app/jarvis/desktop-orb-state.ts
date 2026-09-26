import { atom } from 'nanostores'

import type { Locale } from '@/i18n'
import { persistString, storedString } from '@/lib/storage'

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

export const ORB_WINDOW_SIZE = { width: 320, height: 360 }

/** Preserve the grabbed point; never snap the orb's centre to the pointer. */
export function moveOrb(
  origin: { x: number; y: number },
  start: { x: number; y: number },
  pointer: { x: number; y: number }
) {
  return { x: Math.round(origin.x + pointer.x - start.x), y: Math.round(origin.y + pointer.y - start.y) }
}
