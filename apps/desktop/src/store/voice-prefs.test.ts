import { describe, expect, it, vi } from 'vitest'

vi.mock('@/hermes', () => ({
  getHermesConfigRecord: vi.fn(async () => ({})),
  saveHermesConfig: vi.fn(async () => undefined)
}))

import { $voiceSilenceMs, $voiceStopPhrase, applyVoiceSilenceMsFromConfig, applyVoiceStopPhraseFromConfig } from './voice-prefs'

describe('applyVoiceStopPhraseFromConfig', () => {
  it('defaults to "stop" when the key is absent (backend default applies)', () => {
    applyVoiceStopPhraseFromConfig({ voice: {} })
    expect($voiceStopPhrase.get()).toBe('stop')

    applyVoiceStopPhraseFromConfig(null)
    expect($voiceStopPhrase.get()).toBe('stop')
  })

  it('uses the first configured phrase so a custom phrase renders correctly', () => {
    applyVoiceStopPhraseFromConfig({ voice: { stop_phrases: ['goodbye hermes', 'stop'] } })
    expect($voiceStopPhrase.get()).toBe('goodbye hermes')
  })

  it('coerces a bare string like the backend does', () => {
    applyVoiceStopPhraseFromConfig({ voice: { stop_phrases: 'halt' } })
    expect($voiceStopPhrase.get()).toBe('halt')
  })

  it('null phrase when stop phrases are disabled — no notice is shown', () => {
    applyVoiceStopPhraseFromConfig({ voice: { stop_phrases: [] } })
    expect($voiceStopPhrase.get()).toBeNull()
  })

  it('malformed entries are skipped; all-blank list disables', () => {
    applyVoiceStopPhraseFromConfig({ voice: { stop_phrases: ['  ', ''] } })
    expect($voiceStopPhrase.get()).toBeNull()
  })
})

describe('applyVoiceSilenceMsFromConfig', () => {
  it('converts a configured silence_duration from seconds to milliseconds', () => {
    applyVoiceSilenceMsFromConfig({ voice: { silence_duration: 10 } })
    expect($voiceSilenceMs.get()).toBe(10_000)
  })

  it('falls back to the documented 3s default when the key is absent', () => {
    applyVoiceSilenceMsFromConfig({ voice: {} })
    expect($voiceSilenceMs.get()).toBe(3_000)

    applyVoiceSilenceMsFromConfig(null)
    expect($voiceSilenceMs.get()).toBe(3_000)
  })

  it('falls back to the default for non-numeric or non-positive values (hand-edited config.yaml)', () => {
    applyVoiceSilenceMsFromConfig({ voice: { silence_duration: true } })
    expect($voiceSilenceMs.get()).toBe(3_000)

    applyVoiceSilenceMsFromConfig({ voice: { silence_duration: '5' } })
    expect($voiceSilenceMs.get()).toBe(3_000)

    applyVoiceSilenceMsFromConfig({ voice: { silence_duration: 0 } })
    expect($voiceSilenceMs.get()).toBe(3_000)

    applyVoiceSilenceMsFromConfig({ voice: { silence_duration: -1 } })
    expect($voiceSilenceMs.get()).toBe(3_000)
  })
})
