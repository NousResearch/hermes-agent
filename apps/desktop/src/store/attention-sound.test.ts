import { describe, expect, it } from 'vitest'

import {
  $attentionSoundVariantId,
  DEFAULT_ATTENTION_SOUND_VARIANT_ID,
  resolveAttentionSoundVariantId,
  setAttentionSoundVariantId
} from './attention-sound'
import { DEFAULT_COMPLETION_SOUND_VARIANT_ID } from './completion-sound'

// The whole point of two rows in Settings → Notifications → Sounds is that the
// cues stay distinguishable: "answer me" must never sound like "turn done".
describe('attention sound preset', () => {
  it('defaults to a different preset than the completion cue', () => {
    expect(DEFAULT_ATTENTION_SOUND_VARIANT_ID).not.toBe(DEFAULT_COMPLETION_SOUND_VARIANT_ID)
    expect(resolveAttentionSoundVariantId(DEFAULT_ATTENTION_SOUND_VARIANT_ID)).toBe(DEFAULT_ATTENTION_SOUND_VARIANT_ID)
  })

  it('keeps in-range presets and snaps anything else back to the default', () => {
    expect(resolveAttentionSoundVariantId(7)).toBe(7)
    expect(resolveAttentionSoundVariantId(14)).toBe(14)

    for (const invalid of [0, 15, -3, 2.5, Number.NaN]) {
      expect(resolveAttentionSoundVariantId(invalid)).toBe(DEFAULT_ATTENTION_SOUND_VARIANT_ID)
    }
  })

  it('applies through the atom, so the settings select and the cue agree', () => {
    setAttentionSoundVariantId(9)
    expect($attentionSoundVariantId.get()).toBe(9)

    setAttentionSoundVariantId(99)
    expect($attentionSoundVariantId.get()).toBe(DEFAULT_ATTENTION_SOUND_VARIANT_ID)
  })
})
