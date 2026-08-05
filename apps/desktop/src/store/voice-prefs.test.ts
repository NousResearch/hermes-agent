import { describe, expect, it, vi } from 'vitest'

vi.mock('@/hermes', () => ({
  getHermesConfigRecord: vi.fn(async () => ({})),
  saveHermesConfig: vi.fn(async () => undefined)
}))

import {
  $voiceFollowUpIdleSeconds,
  $voiceStopPhrase,
  applyFollowUpIdleFromConfig,
  applyVoiceStopPhraseFromConfig,
  DEFAULT_FOLLOW_UP_IDLE_SECONDS,
  parseFollowUpIdleSeconds
} from './voice-prefs'

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

describe('parseFollowUpIdleSeconds / applyFollowUpIdleFromConfig', () => {
  it('defaults to 60 seconds when the key is absent', () => {
    expect(parseFollowUpIdleSeconds(undefined)).toBe(DEFAULT_FOLLOW_UP_IDLE_SECONDS)
    applyFollowUpIdleFromConfig({ voice: {} })
    expect($voiceFollowUpIdleSeconds.get()).toBe(DEFAULT_FOLLOW_UP_IDLE_SECONDS)
  })

  it('accepts 0 as the legacy always-rearm mode', () => {
    expect(parseFollowUpIdleSeconds(0)).toBe(0)
    applyFollowUpIdleFromConfig({ voice: { follow_up_idle_seconds: 0 } })
    expect($voiceFollowUpIdleSeconds.get()).toBe(0)
  })

  it('accepts a custom positive window', () => {
    applyFollowUpIdleFromConfig({ voice: { follow_up_idle_seconds: 90 } })
    expect($voiceFollowUpIdleSeconds.get()).toBe(90)
  })

  it('falls back to the default for negative or non-numeric values', () => {
    expect(parseFollowUpIdleSeconds(-1)).toBe(DEFAULT_FOLLOW_UP_IDLE_SECONDS)
    expect(parseFollowUpIdleSeconds('nope')).toBe(DEFAULT_FOLLOW_UP_IDLE_SECONDS)
    expect(parseFollowUpIdleSeconds(Number.NaN)).toBe(DEFAULT_FOLLOW_UP_IDLE_SECONDS)
  })
})
