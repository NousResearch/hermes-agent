import { describe, expect, it } from 'vitest'

import { FREE_INPUT_KEYS, SECTIONS } from './constants'
import { voiceProviderKeys } from './voice-provider-fields'

const voiceKeys = SECTIONS.find(s => s.id === 'voice')?.keys ?? []

describe('voiceProviderKeys', () => {
  it('covers every built-in TTS provider the Capabilities picker offers', () => {
    // Every provider key the backend TOOL_CATEGORIES["tts"] rows can carry
    // (tts_provider values) must resolve to at least one config field, so the
    // Capabilities panel never renders a silently-empty settings block.
    for (const provider of [
      'edge',
      'openai',
      'xai',
      'elevenlabs',
      'mistral',
      'gemini',
      'kittentts',
      'piper',
      'deepinfra',
      'minimax'
    ]) {
      expect(voiceProviderKeys('tts', provider).length, provider).toBeGreaterThan(0)
    }
  })

  it('scopes to the exact provider segment (no prefix bleed)', () => {
    expect(voiceProviderKeys('tts', 'mini')).toEqual([])
    expect(voiceProviderKeys('stt', 'openai')).toEqual(['stt.openai.model'])
  })
})

describe('voice field option coverage', () => {
  it('every free-input voice key that lives in the Voice section has suggestions or is intentionally bare', () => {
    // Free-input keys don't *require* ENUM_OPTIONS (an empty datalist is
    // fine), but any that do declare options must be actual Voice-section
    // fields — a typo'd key here would silently do nothing.
    for (const key of FREE_INPUT_KEYS) {
      expect(voiceKeys, key).toContain(key)
    }
  })
})
