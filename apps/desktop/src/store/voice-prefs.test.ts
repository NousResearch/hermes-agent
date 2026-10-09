import { describe, expect, it, vi } from 'vitest'

vi.mock('@/hermes', () => ({
  getHermesConfigRecord: vi.fn(async () => ({})),
  saveHermesConfig: vi.fn(async () => undefined)
}))

import { saveHermesConfig } from '@/hermes'
import { isVoiceStopCommand } from '@/lib/voice-stop-word'

import {
  $bargeInEnabled,
  $bargeInThresholdMultiplier,
  $voiceSilenceMs,
  $voiceStopPhrase,
  $voiceStopPhraseConfig,
  applyBargeInEnabledFromConfig,
  applyBargeInThresholdFromConfig,
  applyVoiceSilenceMsFromConfig,
  applyVoiceStopPhraseFromConfig
} from './voice-prefs'

// jsdom's Storage keeps setItem on Storage.prototype, and a spy installed on the instance
// never intercepts the call (Node 22/24). The Node 26 shim in vitest.setup.ts defines
// setItem on the object itself, so spy wherever the method actually lives.
function spyOnStorageWrite() {
  return vi.spyOn(Object.hasOwn(localStorage, 'setItem') ? localStorage : Storage.prototype, 'setItem')
}

it('keeps the desktop toggle local across config refreshes', async () => {
  for (const fails of [false, true]) {
    for (const enabled of [false, true]) {
      localStorage.clear()
      vi.resetModules()
      const prefs = await import('./voice-prefs')
      const write = spyOnStorageWrite()

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
      const write = spyOnStorageWrite()

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

// `voice.barge_in` mirrors the gateway's `_arm_barge_listener_if_enabled`
// (tui_gateway/methods_voice.py): the listener is armed unless the key is
// explicitly false.
describe('applyBargeInEnabledFromConfig', () => {
  it('disarms only an explicit false', () => {
    applyBargeInEnabledFromConfig({ voice: { barge_in: false } })
    expect($bargeInEnabled.get()).toBe(false)

    applyBargeInEnabledFromConfig({ voice: { barge_in: true } })
    expect($bargeInEnabled.get()).toBe(true)
  })

  it('absent, null, or malformed values keep barge-in enabled', () => {
    for (const voice of [undefined, {}, { barge_in: null }, { barge_in: 'nope' }, { barge_in: 0 }]) {
      applyBargeInEnabledFromConfig({ voice: { barge_in: false } })
      applyBargeInEnabledFromConfig({ voice })
      expect($bargeInEnabled.get()).toBe(true)
    }

    applyBargeInEnabledFromConfig(null)
    expect($bargeInEnabled.get()).toBe(true)
  })
})

// `voice.silence_duration` drives the desktop loop the way it drives the
// CLI/TUI capture paths, but only when the user actually changed it: `/api/config`
// merges DEFAULT_CONFIG, so an untouched install reports the backend default
// (3.0) rather than omitting the key, and reading that unconditionally would
// triple the hold for everyone (the loop was tuned to 1.25 s).
describe('applyVoiceSilenceMsFromConfig', () => {
  const backendDefault = { voice: { silence_duration: 3.0 } }

  it('a user-set silence_duration overrides the desktop hold (seconds to ms)', () => {
    applyVoiceSilenceMsFromConfig({ voice: { silence_duration: 0.7 } }, backendDefault)
    expect($voiceSilenceMs.get()).toBe(700)

    applyVoiceSilenceMsFromConfig({ voice: { silence_duration: 10 } }, backendDefault)
    expect($voiceSilenceMs.get()).toBe(10_000)

    applyVoiceSilenceMsFromConfig({ voice: { silence_duration: '2' } }, backendDefault)
    expect($voiceSilenceMs.get()).toBe(2_000)
  })

  it('an untouched install keeps the tuned 1.25 s desktop hold', () => {
    applyVoiceSilenceMsFromConfig(backendDefault, backendDefault)
    expect($voiceSilenceMs.get()).toBe(1_250)

    // Defaults endpoint unavailable: the backend default is still recognisable.
    applyVoiceSilenceMsFromConfig(backendDefault, {})
    expect($voiceSilenceMs.get()).toBe(1_250)

    applyVoiceSilenceMsFromConfig({ voice: {} }, backendDefault)
    expect($voiceSilenceMs.get()).toBe(1_250)

    applyVoiceSilenceMsFromConfig(null)
    expect($voiceSilenceMs.get()).toBe(1_250)
  })

  it('malformed or non-positive values keep the default like the gateway lookup', () => {
    for (const raw of [0, -1, true, 'quiet', null, {}]) {
      applyVoiceSilenceMsFromConfig({ voice: { silence_duration: 0.7 } }, backendDefault)
      applyVoiceSilenceMsFromConfig({ voice: { silence_duration: raw } }, backendDefault)
      expect($voiceSilenceMs.get()).toBe(1_250)
    }
  })
})
