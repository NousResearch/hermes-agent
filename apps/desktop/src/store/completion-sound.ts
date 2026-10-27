import { atom } from 'nanostores'

import { persistStore, persistString, storedString } from '@/lib/storage'

const STORAGE_KEY = 'hermes.desktop.completionSoundVariantId'

export const DEFAULT_COMPLETION_SOUND_VARIANT_ID = 1

// "None": variant 0 disables the turn-end cue but stays a selectable settings value.
export const NO_COMPLETION_SOUND_VARIANT_ID = 0

// Range mirrors COMPLETION_SOUND_VARIANTS in lib/completion-sound.ts (0 = no
// sound). Validating by range (not membership) keeps this store free of a
// dependency on the lib, which imports the atom back — a membership check would
// close that cycle.
const VARIANT_COUNT = 14

export function resolveCompletionSoundVariantId(variantId: number): number {
  return Number.isInteger(variantId) && variantId >= 0 && variantId <= VARIANT_COUNT
    ? variantId
    : DEFAULT_COMPLETION_SOUND_VARIANT_ID
}

function load(): number {
  const stored = storedString(STORAGE_KEY)

  return stored ? resolveCompletionSoundVariantId(Number.parseInt(stored, 10)) : DEFAULT_COMPLETION_SOUND_VARIANT_ID
}

export const $completionSoundVariantId = atom(load())

persistStore($completionSoundVariantId, (id, options) => persistString(STORAGE_KEY, String(id), options))

export function setCompletionSoundVariantId(variantId: number) {
  $completionSoundVariantId.set(resolveCompletionSoundVariantId(variantId))
}
