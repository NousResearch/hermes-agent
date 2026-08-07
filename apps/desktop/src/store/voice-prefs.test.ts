import { describe, expect, it, vi } from 'vitest'

vi.mock('@/hermes', () => ({
  getHermesConfigRecord: vi.fn(async () => ({})),
  saveHermesConfig: vi.fn(async () => undefined)
}))

import {
  $voiceConversationIdleTimeoutMs,
  $voiceStopPhrase,
  applyVoiceConversationIdleTimeoutFromConfig,
  applyVoiceStopPhraseFromConfig
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

describe('applyVoiceConversationIdleTimeoutFromConfig', () => {
  it('converts the configured inactivity limit from seconds to milliseconds', () => {
    applyVoiceConversationIdleTimeoutFromConfig({ voice: { conversation_idle_timeout_seconds: 600 } })
    expect($voiceConversationIdleTimeoutMs.get()).toBe(600_000)
  })

  it.each([undefined, null, {}, { voice: {} }])('uses the backend default when absent (%j)', config => {
    applyVoiceConversationIdleTimeoutFromConfig(config)
    expect($voiceConversationIdleTimeoutMs.get()).toBe(600_000)
  })

  it.each([-1, Number.NaN, Number.POSITIVE_INFINITY, '30', null])(
    'uses the backend default for malformed values (%j)',
    value => {
      applyVoiceConversationIdleTimeoutFromConfig({ voice: { conversation_idle_timeout_seconds: value } })
      expect($voiceConversationIdleTimeoutMs.get()).toBe(600_000)
    }
  )

  it('uses the backend default when the configured delay exceeds the browser timer range', () => {
    applyVoiceConversationIdleTimeoutFromConfig({
      voice: { conversation_idle_timeout_seconds: Number.MAX_SAFE_INTEGER }
    })

    expect($voiceConversationIdleTimeoutMs.get()).toBe(600_000)
  })
})
