import { atom } from 'nanostores'

import { persistString, storedString } from '@/lib/storage'

const STORAGE_KEY = 'hermes.desktop.completionSoundVariantId'
const VOLUME_STORAGE_KEY = 'hermes.desktop.completionSoundVolume'

export const DEFAULT_COMPLETION_SOUND_VARIANT_ID = 1

// Default loudness as a linear multiplier applied on top of the baked-in
// synthesis gains. 1 = the level the sounds were originally designed at
// (master 0.48 → dry 0.88 ≈ audible but quiet). 0.5 halves it, 2 doubles it.
export const DEFAULT_COMPLETION_SOUND_VOLUME = 2

// Range mirrors COMPLETION_SOUND_VARIANTS in lib/completion-sound.ts. Validating
// by range (not membership) keeps this store free of a dependency on the lib,
// which imports the atom back — a membership check would close that cycle.
const VARIANT_COUNT = 15

export function resolveCompletionSoundVariantId(variantId: number): number {
  return Number.isInteger(variantId) && variantId >= 1 && variantId <= VARIANT_COUNT
    ? variantId
    : DEFAULT_COMPLETION_SOUND_VARIANT_ID
}

export function resolveCompletionSoundVolume(value: number): number {
  // Clamp to a sane audible range: 0 (mute) … 4× the designed level.
  if (!Number.isFinite(value)) {
    return DEFAULT_COMPLETION_SOUND_VOLUME
  }

  return Math.min(4, Math.max(0, value))
}

function load(): number {
  const stored = storedString(STORAGE_KEY)

  return stored ? resolveCompletionSoundVariantId(Number.parseInt(stored, 10)) : DEFAULT_COMPLETION_SOUND_VARIANT_ID
}

function loadVolume(): number {
  const stored = storedString(VOLUME_STORAGE_KEY)

  return stored ? resolveCompletionSoundVolume(Number.parseFloat(stored)) : DEFAULT_COMPLETION_SOUND_VOLUME
}

export const $completionSoundVariantId = atom(load())

$completionSoundVariantId.subscribe(id => persistString(STORAGE_KEY, String(id)))

export function setCompletionSoundVariantId(variantId: number) {
  $completionSoundVariantId.set(resolveCompletionSoundVariantId(variantId))
}

export const $completionSoundVolume = atom(loadVolume())

$completionSoundVolume.subscribe(volume => persistString(VOLUME_STORAGE_KEY, String(volume)))

export function setCompletionSoundVolume(volume: number) {
  $completionSoundVolume.set(resolveCompletionSoundVolume(volume))
}
