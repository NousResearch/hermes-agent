import { describe, expect, it, vi } from 'vitest'

vi.mock('@/hermes', () => ({
  getHermesConfigRecord: vi.fn(async () => ({})),
  saveHermesConfig: vi.fn(async () => undefined)
}))

import { saveHermesConfig } from '@/hermes'

import { $voiceStopPhrase, $voiceStopPhrases, applyVoiceStopPhraseFromConfig } from './voice-prefs'

it('keeps the desktop toggle local across config refreshes', async () => {
  for (const fails of [false, true]) {
    for (const enabled of [false, true]) {
      localStorage.clear()
      vi.resetModules()
      const prefs = await import('./voice-prefs')
      const write = vi.spyOn(Storage.prototype, 'setItem')

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
      const write = vi.spyOn(Storage.prototype, 'setItem')

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
  it('keeps the hint and the complete matcher list on the same parsed configuration', () => {
    for (const [raw, expected] of [
      [undefined, ['stop']],
      [null, ['stop']],
      [{ phrase: 'halt' }, ['stop']],
      [' halt ', ['halt']],
      [
        ['  ', {}, null, ' 그만 ', '대화 종료', 123],
        ['그만', '대화 종료', '123']
      ],
      [[], []]
    ] as const) {
      applyVoiceStopPhraseFromConfig({ voice: { stop_phrases: raw } })
      expect($voiceStopPhrases.get()).toEqual(expected)
      expect($voiceStopPhrase.get()).toBe(expected[0] ?? null)
    }
  })

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
