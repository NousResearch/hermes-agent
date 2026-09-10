import { describe, expect, it } from 'vitest'

import { $ttsConclusionGraceMs, $ttsConclusionOnly, applyTtsConclusionFromConfig } from './voice-prefs'

describe('applyTtsConclusionFromConfig', () => {
  it('stays off unless the config explicitly enables it', () => {
    applyTtsConclusionFromConfig(null)
    expect($ttsConclusionOnly.get()).toBe(false)
    expect($ttsConclusionGraceMs.get()).toBe(1500)

    applyTtsConclusionFromConfig({ voice: { tts_conclusion_grace_ms: 2500 } })
    expect($ttsConclusionOnly.get()).toBe(false)
    expect($ttsConclusionGraceMs.get()).toBe(2500)

    applyTtsConclusionFromConfig({ voice: { tts_conclusion_only: true, tts_conclusion_grace_ms: 800 } })
    expect($ttsConclusionOnly.get()).toBe(true)
    expect($ttsConclusionGraceMs.get()).toBe(800)
  })

  it('ignores non-numeric grace windows instead of poisoning the timer', () => {
    $ttsConclusionGraceMs.set(1200)
    applyTtsConclusionFromConfig({ voice: { tts_conclusion_grace_ms: 'soon' } })
    expect($ttsConclusionGraceMs.get()).toBe(1200)

    applyTtsConclusionFromConfig({ voice: { tts_conclusion_grace_ms: -5 } })
    expect($ttsConclusionGraceMs.get()).toBe(1200)
  })
})
