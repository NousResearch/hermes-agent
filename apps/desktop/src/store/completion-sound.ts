import { atom } from 'nanostores'

import { persistString, storedString } from '@/lib/storage'

const STORAGE_KEY = 'hermes.desktop.completionSoundVariantId'
const VOLUME_STORAGE_KEY = 'hermes.desktop.completionSoundVolume'

export const DEFAULT_COMPLETION_SOUND_VARIANT_ID = 1

// Range mirrors COMPLETION_SOUND_VARIANTS in lib/completion-sound.ts. Validating
// by range (not membership) keeps this store free of a dependency on the lib,
// which imports the atom back — a membership check would close that cycle.
const VARIANT_COUNT = 14

export function resolveCompletionSoundVariantId(variantId: number): number {
  return Number.isInteger(variantId) && variantId >= 1 && variantId <= VARIANT_COUNT
    ? variantId
    : DEFAULT_COMPLETION_SOUND_VARIANT_ID
}

function load(): number {
  const stored = storedString(STORAGE_KEY)

  return stored ? resolveCompletionSoundVariantId(Number.parseInt(stored, 10)) : DEFAULT_COMPLETION_SOUND_VARIANT_ID
}

export const $completionSoundVariantId = atom(load())

$completionSoundVariantId.subscribe(id => persistString(STORAGE_KEY, String(id)))

export function setCompletionSoundVariantId(variantId: number) {
  $completionSoundVariantId.set(resolveCompletionSoundVariantId(variantId))
}

export const COMPLETION_SOUND_VOLUME_MIN = 0
export const COMPLETION_SOUND_VOLUME_MAX = 100
// The loudest position is the default: the chime has no headroom above it (see
// lib/completion-sound), so the slider only turns it down.
export const DEFAULT_COMPLETION_SOUND_VOLUME = COMPLETION_SOUND_VOLUME_MAX

export function resolveCompletionSoundVolume(volume: number): number {
  return Number.isFinite(volume)
    ? Math.min(COMPLETION_SOUND_VOLUME_MAX, Math.max(COMPLETION_SOUND_VOLUME_MIN, Math.round(volume)))
    : DEFAULT_COMPLETION_SOUND_VOLUME
}

function loadVolume(): number {
  const stored = storedString(VOLUME_STORAGE_KEY)

  return stored ? resolveCompletionSoundVolume(Number.parseInt(stored, 10)) : DEFAULT_COMPLETION_SOUND_VOLUME
}

// 0–100 slider position, not a gain: lib/completion-sound maps it onto a dB curve.
export const $completionSoundVolume = atom(loadVolume())

$completionSoundVolume.subscribe(volume => persistString(VOLUME_STORAGE_KEY, String(volume)))

export function setCompletionSoundVolume(volume: number) {
  $completionSoundVolume.set(resolveCompletionSoundVolume(volume))
}
