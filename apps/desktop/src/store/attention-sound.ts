import { atom } from 'nanostores'

import { persistString, storedString } from '@/lib/storage'

// The attention cue's preset, kept separate from the completion cue's: "answer
// me" and "turn finished" must never be able to sound alike. Same bank of
// variants as `lib/completion-sound.ts` (COMPLETION_SOUND_VARIANTS); the range
// mirror is validation-only, so this store stays free of a dependency on the
// lib, which imports the atom back — a membership check would close that cycle.
const STORAGE_KEY = 'hermes.desktop.attentionSoundVariantId'

/** Tri-tone message: bright, three notes, nothing like the completion default. */
export const DEFAULT_ATTENTION_SOUND_VARIANT_ID = 4

const VARIANT_COUNT = 14

export function resolveAttentionSoundVariantId(variantId: number): number {
  return Number.isInteger(variantId) && variantId >= 1 && variantId <= VARIANT_COUNT
    ? variantId
    : DEFAULT_ATTENTION_SOUND_VARIANT_ID
}

function load(): number {
  const stored = storedString(STORAGE_KEY)

  return stored ? resolveAttentionSoundVariantId(Number.parseInt(stored, 10)) : DEFAULT_ATTENTION_SOUND_VARIANT_ID
}

export const $attentionSoundVariantId = atom(load())

$attentionSoundVariantId.subscribe(id => persistString(STORAGE_KEY, String(id)))

export function setAttentionSoundVariantId(variantId: number) {
  $attentionSoundVariantId.set(resolveAttentionSoundVariantId(variantId))
}
