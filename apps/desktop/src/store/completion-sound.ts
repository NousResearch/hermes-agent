import { atom } from 'nanostores'

import { persistString, storedString } from '@/lib/storage'

const STORAGE_KEY = 'hermes.desktop.completionSoundVariantId'
const LEGACY_DEFAULT_MIGRATED_KEY = 'hermes.desktop.completionSoundLegacyDefaultMigrated'

export const OFF_COMPLETION_SOUND_VARIANT_ID = 0
export const DEFAULT_COMPLETION_SOUND_VARIANT_ID = OFF_COMPLETION_SOUND_VARIANT_ID

// Range mirrors COMPLETION_SOUND_VARIANTS in lib/completion-sound.ts. Validating
// by range (not membership) keeps this store free of a dependency on the lib,
// which imports the atom back — a membership check would close that cycle.
const VARIANT_COUNT = 14

export function resolveCompletionSoundVariantId(variantId: number): number {
  return Number.isInteger(variantId) && variantId >= OFF_COMPLETION_SOUND_VARIANT_ID && variantId <= VARIANT_COUNT
    ? variantId
    : DEFAULT_COMPLETION_SOUND_VARIANT_ID
}

function load(): number {
  const stored = storedString(STORAGE_KEY)

  if (stored === '1' && storedString(LEGACY_DEFAULT_MIGRATED_KEY) !== 'true') {
    persistString(LEGACY_DEFAULT_MIGRATED_KEY, 'true')

    return OFF_COMPLETION_SOUND_VARIANT_ID
  }

  return stored ? resolveCompletionSoundVariantId(Number.parseInt(stored, 10)) : DEFAULT_COMPLETION_SOUND_VARIANT_ID
}

export const $completionSoundVariantId = atom(load())

$completionSoundVariantId.subscribe(id => persistString(STORAGE_KEY, String(id)))

export function setCompletionSoundVariantId(variantId: number) {
  $completionSoundVariantId.set(resolveCompletionSoundVariantId(variantId))
}
