import { atom } from 'nanostores'

import { persistString, storedString } from '@/lib/storage'

const STORAGE_KEY = 'hermes.desktop.completionSoundVolume'

/**
 * User-facing loudness multiplier for the turn-end cues, on top of the bank's own
 * base gain. 0 = silent, 1 = the shipped default, 2 = double the default gain.
 */
export const MIN_COMPLETION_SOUND_VOLUME = 0
export const MAX_COMPLETION_SOUND_VOLUME = 2
export const DEFAULT_COMPLETION_SOUND_VOLUME = 1

// Clamping lives here rather than in lib/completion-sound.ts: the lib imports this
// atom, so a dependency the other way would close the cycle.
export function resolveCompletionSoundVolume(volume: number): number {
  if (!Number.isFinite(volume)) {
    return DEFAULT_COMPLETION_SOUND_VOLUME
  }

  return Math.min(MAX_COMPLETION_SOUND_VOLUME, Math.max(MIN_COMPLETION_SOUND_VOLUME, volume))
}

function load(): number {
  const stored = storedString(STORAGE_KEY)

  return stored ? resolveCompletionSoundVolume(Number.parseFloat(stored)) : DEFAULT_COMPLETION_SOUND_VOLUME
}

export const $completionSoundVolume = atom(load())

$completionSoundVolume.subscribe(volume => persistString(STORAGE_KEY, String(volume)))

export function setCompletionSoundVolume(volume: number) {
  $completionSoundVolume.set(resolveCompletionSoundVolume(volume))
}
