import { describe, expect, it, vi } from 'vitest'

vi.mock('@/hermes', () => ({
  getHermesConfigRecord: vi.fn(async () => ({})),
  saveHermesConfig: vi.fn(async () => undefined)
}))

import { saveHermesConfig } from '@/hermes'
import { isVoiceStopCommand } from '@/lib/voice-stop-word'

import {
  $bargeInThresholdMultiplier,
  $voiceStopPhrase,
  $voiceStopPhraseConfig,
  applyBargeInThresholdFromConfig,
  applyVoiceStopPhraseFromConfig
} from './voice-prefs'

it('keeps the desktop toggle local across config refreshes', async () => {
  for (const fails of [false, true]) {
    for (const enabled of [false, true]) {
      localStorage.clear()
      vi.resetModules()
      const prefs = await import('./voice-prefs')
      const write = vi.spyOn(localStorage, 'setItem')

      if (fails) {
        write.mockImplementation(() => {
          throw new DOMException('Full', 'QuotaExceededError')
        })
      }

      vi.mocked(saveHermesConfig).mockClear()

      try {
        await prefs.setAutoSpeakReplies(enabled)
        prefs.applyAutoSpeakFromConfig({ voice: { auto_tts: !enabled } })
        expect(prefs.$autoSpeakReplies.get()).toBe(enabled)
        expect(saveHermesConfig).not.toHaveBeenCalled()
        expect(localStorage.getItem('hermes.desktop.autoSpeakReplies')).toBe(fails ? null : String(enabled))
      } finally {
        write.mockRestore()
      }
    }
  }
})

it('migrates the legacy preference once, not on every refresh', async () => {
  for (const fails of [false, true]) {
    for (const enabled of [false, true]) {
      localStorage.clear()
      vi.resetModules()
      const prefs = await import('./voice-prefs')
      const write = vi.spyOn(localStorage, 'setItem')

      if (fails) {
        write.mockImplementation(() => {
          throw new DOMException('Denied', 'SecurityError')
        })
      }

      try {
        prefs.applyAutoSpeakFromConfig(null)
        expect(localStorage.getItem('hermes.desktop.autoSpeakReplies')).toBeNull()
        prefs.applyAutoSpeakFromConfig({ voice: { auto_tts: enabled } })
        prefs.applyAutoSpeakFromConfig({ voice: { auto_tts: !enabled } })
        expect(prefs.$autoSpeakReplies.get()).toBe(enabled)
        expect(localStorage.getItem('hermes.desktop.autoSpeakReplies')).toBe(fails ? null : String(enabled))
      } finally {
        write.mockRestore()
      }
    }
  }
})

describe('applyVoiceStopPhraseFromConfig', () => {
  it('defaults to "stop" when the key is absent (backend default applies)', () => {
    applyVoiceStopPhraseFromConfig({ voice: {} })
    expect($voiceStopPhrase.get()).toBe('stop')
    expect($voiceStopPhraseConfig.get()).toEqual({ mode: 'default' })

    applyVoiceStopPhraseFromConfig(null)
    expect($voiceStopPhrase.get()).toBe('stop')
    expect($voiceStopPhraseConfig.get()).toEqual({ mode: 'default' })
  })

  it('uses the first configured phrase so a custom phrase renders correctly', () => {
    applyVoiceStopPhraseFromConfig({ voice: { stop_phrases: ['goodbye hermes', 'stop'] } })
    expect($voiceStopPhrase.get()).toBe('goodbye hermes')
    expect($voiceStopPhraseConfig.get()).toEqual({
      mode: 'custom',
      phrases: ['goodbye hermes', 'stop']
    })
  })

  it('coerces a bare string like the backend does', () => {
    applyVoiceStopPhraseFromConfig({ voice: { stop_phrases: 'halt' } })
    expect($voiceStopPhrase.get()).toBe('halt')
    expect($voiceStopPhraseConfig.get()).toEqual({ mode: 'custom', phrases: ['halt'] })
  })

  it('null phrase when stop phrases are disabled — no notice is shown', () => {
    applyVoiceStopPhraseFromConfig({ voice: { stop_phrases: [] } })
    expect($voiceStopPhrase.get()).toBeNull()
    expect($voiceStopPhraseConfig.get()).toEqual({ mode: 'disabled' })
  })

  it('malformed entries are skipped; all-blank list disables', () => {
    applyVoiceStopPhraseFromConfig({ voice: { stop_phrases: ['  ', ''] } })
    expect($voiceStopPhrase.get()).toBeNull()
    expect($voiceStopPhraseConfig.get()).toEqual({ mode: 'disabled' })
  })
})

// The live matcher reads the atoms these seed, so drive it through them the way
// `useHermesConfig` does: `/api/config` (defaults merged in) + `/api/config/defaults`.
describe('spoken stop follows the loaded voice.stop_phrases (#117801)', () => {
  const defaults = { voice: { stop_phrases: ['stop'] } }

  it('a configured Russian list ends the chat and replaces the English list', () => {
    applyVoiceStopPhraseFromConfig({ voice: { stop_phrases: ['отбой', 'стоп', 'stop'] } }, defaults)

    expect(isVoiceStopCommand('Отбой.')).toBe(true)
    expect(isVoiceStopCommand('стоп')).toBe(true)
    expect(isVoiceStopCommand('goodbye')).toBe(false)
  })

  it('an untouched install (merged backend default) keeps the built-in English list', () => {
    applyVoiceStopPhraseFromConfig({ voice: { stop_phrases: ['stop'] } }, defaults)

    expect($voiceStopPhrase.get()).toBe('stop')
    expect(isVoiceStopCommand('goodbye')).toBe(true)
    expect(isVoiceStopCommand('never mind')).toBe(true)

    // Defaults endpoint unavailable: still recognised as the backend default.
    applyVoiceStopPhraseFromConfig({ voice: { stop_phrases: ['stop'] } }, {})
    expect(isVoiceStopCommand('goodbye')).toBe(true)
  })

  it('a malformed value falls back to the default like the backend', () => {
    applyVoiceStopPhraseFromConfig({ voice: { stop_phrases: { ru: 'отбой' } } }, defaults)

    expect($voiceStopPhrase.get()).toBe('stop')
    expect(isVoiceStopCommand('stop')).toBe(true)
  })

  it('an empty list disables spoken stop', () => {
    applyVoiceStopPhraseFromConfig({ voice: { stop_phrases: [] } }, defaults)

    expect(isVoiceStopCommand('stop')).toBe(false)
  })
})

describe('applyBargeInThresholdFromConfig', () => {
  it('adopts a positive voice.barge_in_threshold_multiplier', () => {
    applyBargeInThresholdFromConfig({ voice: { barge_in_threshold_multiplier: 1.5 } })
    expect($bargeInThresholdMultiplier.get()).toBe(1.5)

    applyBargeInThresholdFromConfig({ voice: { barge_in_threshold_multiplier: '2' } })
    expect($bargeInThresholdMultiplier.get()).toBe(2)
  })

  it('unset, zero, or malformed values leave the stock sensitivity', () => {
    for (const voice of [{}, { barge_in_threshold_multiplier: 0 }, { barge_in_threshold_multiplier: 'loud' }]) {
      applyBargeInThresholdFromConfig({ voice: { barge_in_threshold_multiplier: 1.5 } })
      applyBargeInThresholdFromConfig({ voice })
      expect($bargeInThresholdMultiplier.get()).toBeNull()
    }

    applyBargeInThresholdFromConfig(null)
    expect($bargeInThresholdMultiplier.get()).toBeNull()
  })
})
