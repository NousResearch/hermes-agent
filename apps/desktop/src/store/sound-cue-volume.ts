import { atom } from 'nanostores'

import { persistString, storedString } from '@/lib/storage'

const STORAGE_KEY = 'hermes.desktop.soundCueVolume'

// 1 = the original shipped loudness. The synthesized turn-end and wake
// chimes (lib/completion-sound.ts, lib/wake-sound.ts) were tuned quiet by
// default; this lets users turn them up well past that without touching
// macOS's system/alert volume, which most people don't want to disturb for
// one app's cues. Capped at 6x: a limiter on the output bus (see
// completion-sound.ts / wake-sound.ts) keeps that genuinely loud without
// harsh digital clipping.
export const DEFAULT_SOUND_CUE_VOLUME = 1
export const SOUND_CUE_VOLUME_MIN = 0
export const SOUND_CUE_VOLUME_MAX = 6

export function resolveSoundCueVolume(volume: number): number {
  return Number.isFinite(volume)
    ? Math.min(Math.max(volume, SOUND_CUE_VOLUME_MIN), SOUND_CUE_VOLUME_MAX)
    : DEFAULT_SOUND_CUE_VOLUME
}

function load(): number {
  const stored = storedString(STORAGE_KEY)

  return stored ? resolveSoundCueVolume(Number.parseFloat(stored)) : DEFAULT_SOUND_CUE_VOLUME
}

export const $soundCueVolume = atom(load())

$soundCueVolume.subscribe(volume => persistString(STORAGE_KEY, String(volume)))

export function setSoundCueVolume(volume: number) {
  $soundCueVolume.set(resolveSoundCueVolume(volume))
}
