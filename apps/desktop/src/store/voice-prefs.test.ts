import { beforeEach, describe, expect, it } from 'vitest'

import {
  $autoSpeakReplies,
  $voiceStopPhrase,
  applyAutoSpeakFromConfig,
  applyVoiceStopPhraseFromConfig,
  setAutoSpeakReplies
} from './voice-prefs'

describe('desktop-local auto-speak preference', () => {
  beforeEach(() => {
    window.localStorage.clear()
    $autoSpeakReplies.set(false)
  })

  it('migrates from the legacy shared voice.auto_tts value on first run', () => {
    applyAutoSpeakFromConfig({ voice: { auto_tts: true } })

    expect($autoSpeakReplies.get()).toBe(true)
    expect(window.localStorage.getItem('hermes.desktop.autoSpeakReplies')).toBeNull()
  })

  it('keeps the stored local preference over a later config refresh', () => {
    window.localStorage.setItem('hermes.desktop.autoSpeakReplies', 'false')

    applyAutoSpeakFromConfig({ voice: { auto_tts: true } })

    expect($autoSpeakReplies.get()).toBe(false)
  })

  it('toggling persists locally without touching the shared config', async () => {
    await setAutoSpeakReplies(true)

    expect($autoSpeakReplies.get()).toBe(true)
    expect(window.localStorage.getItem('hermes.desktop.autoSpeakReplies')).toBe('true')
  })

  it('a failed local write reverts the atom', async () => {
    // Swap the whole `localStorage` property rather than mocking
    // `Storage.prototype`: under Node 26 the setup file installs a
    // plain-object storage fallback whose methods are own properties (no
    // Storage prototype), while jsdom's Storage wrapper silently drops
    // own-property overrides. Replacing the property works under both.
    const real = window.localStorage

    const failing: Storage = {
      get length() {
        return real.length
      },
      key: index => real.key(index),
      getItem: key => real.getItem(key),
      setItem: () => {
        throw new DOMException('quota exceeded', 'QuotaExceededError')
      },
      removeItem: key => real.removeItem(key),
      clear: () => real.clear()
    }

    Object.defineProperty(window, 'localStorage', { value: failing, configurable: true })

    try {
      await expect(setAutoSpeakReplies(true)).rejects.toThrow()
      expect($autoSpeakReplies.get()).toBe(false)
    } finally {
      Object.defineProperty(window, 'localStorage', { value: real, configurable: true })
    }
  })
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
