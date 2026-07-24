import { atom } from 'nanostores'

import { persistString, storedString } from '@/lib/storage'

const STORAGE_KEY = 'hermes.desktop.completionSoundVariantId'

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

// ── Volume ───────────────────────────────────────────────────────────────────

const VOLUME_STORAGE_KEY = 'hermes.desktop.completionSoundVolume'

/** Default matches the current hardcoded master gain of 0.48 for back-compat. */
export const DEFAULT_COMPLETION_SOUND_VOLUME = 0.48

/** Clamp to [0, 3] so invalid persisted values can't produce extreme gain. */
const VOLUME_MIN = 0
const VOLUME_MAX = 3

export function resolveCompletionSoundVolume(raw: number): number {
  return raw >= VOLUME_MIN && raw <= VOLUME_MAX ? raw : DEFAULT_COMPLETION_SOUND_VOLUME
}

function loadVolume(): number {
  const stored = storedString(VOLUME_STORAGE_KEY)

  if (stored === null) {
    return DEFAULT_COMPLETION_SOUND_VOLUME
  }

  const parsed = Number.parseFloat(stored)

  return Number.isFinite(parsed) ? resolveCompletionSoundVolume(parsed) : DEFAULT_COMPLETION_SOUND_VOLUME
}

export const $completionSoundVolume = atom(loadVolume())

$completionSoundVolume.subscribe(v => persistString(VOLUME_STORAGE_KEY, String(v)))

export function setCompletionSoundVolume(volume: number) {
  $completionSoundVolume.set(resolveCompletionSoundVolume(volume))
}
