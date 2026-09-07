import { describe, expect, it, vi } from 'vitest'

vi.mock('@/hermes', () => ({
  getHermesConfigRecord: vi.fn(async () => ({})),
  saveHermesConfig: vi.fn(async () => undefined)
}))

import { saveHermesConfig } from '@/hermes'

import { $autoSpeakReplies, $voiceStopPhrase, applyAutoSpeakFromConfig, applyVoiceStopPhraseFromConfig, setAutoSpeakReplies } from './voice-prefs'

it('keeps the desktop toggle local across config refreshes', async () => {
  localStorage.clear()
  $autoSpeakReplies.set(false)
  vi.mocked(saveHermesConfig).mockClear()
  await setAutoSpeakReplies(true)
  applyAutoSpeakFromConfig({ voice: { auto_tts: false } })
  expect($autoSpeakReplies.get()).toBe(true)
  expect(saveHermesConfig).not.toHaveBeenCalled()
  expect(localStorage.getItem('hermes.desktop.autoSpeakReplies')).toBe('true')
})

it('migrates the legacy preference once, not on every refresh', () => {
  for (const enabled of [false, true]) {
    localStorage.clear()
    applyAutoSpeakFromConfig(null)
    expect(localStorage.getItem('hermes.desktop.autoSpeakReplies')).toBeNull()
    applyAutoSpeakFromConfig({ voice: { auto_tts: enabled } })
    applyAutoSpeakFromConfig({ voice: { auto_tts: !enabled } })
    expect($autoSpeakReplies.get()).toBe(enabled)
    expect(localStorage.getItem('hermes.desktop.autoSpeakReplies')).toBe(String(enabled))
  }
})

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
