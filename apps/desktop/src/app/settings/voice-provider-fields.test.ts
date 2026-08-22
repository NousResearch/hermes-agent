import { describe, expect, it } from 'vitest'

import { ENUM_OPTIONS, FREE_INPUT_KEYS, SECTIONS } from './constants'
import { detectedConfigKeys, voiceProviderKeys } from './voice-provider-fields'

const voiceKeys = SECTIONS.find(s => s.id === 'voice')?.keys ?? []

describe('voiceProviderKeys', () => {
  it('derives per-provider field keys from the curated Voice section', () => {
    expect(voiceProviderKeys('tts', 'openai')).toEqual(['tts.openai.model', 'tts.openai.voice'])
    expect(voiceProviderKeys('tts', 'elevenlabs')).toEqual(['tts.elevenlabs.voice_id', 'tts.elevenlabs.model_id'])
    expect(voiceProviderKeys('tts', 'edge')).toEqual(['tts.edge.voice'])
  })

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

  it('discovers plugin fields from backend schema and config presence', () => {
    expect(
      voiceProviderKeys(
        'stt',
        'bifrost',
        { 'stt.bifrost.model': { type: 'select' } },
        { stt: { bifrost: { language: 'ru' } } }
      )
    ).toEqual(['stt.bifrost.model', 'stt.bifrost.language'])
  })

  it('marks config-presence-only keys as detected (no schema + no curated entry)', () => {
    expect(
      detectedConfigKeys(
        'stt',
        'bifrost',
        { 'stt.bifrost.model': { type: 'select' } },
        { stt: { bifrost: { language: 'ru', model: 'whisper' } } }
      )
    ).toEqual(['stt.bifrost.language'])
  })

  it('excludes declared (curated/schema) keys from the detected set', () => {
    // tts.openai.model is a curated Voice key; a config value for it must NOT
    // be flagged as detected by presence alone.
    expect(
      detectedConfigKeys('tts', 'openai', {}, { tts: { openai: { model: 'gpt-4o-mini-tts' } } })
    ).toEqual([])
  })
})

describe('voice field option coverage', () => {
  it('offers the current gpt-4o-mini-tts voice set, not just the tts-1 six', () => {
    const voices = ENUM_OPTIONS['tts.openai.voice']

    for (const voice of ['alloy', 'ash', 'ballad', 'cedar', 'coral', 'marin', 'sage', 'verse', 'shimmer']) {
      expect(voices).toContain(voice)
    }
  })

  it('keeps voice/model name fields free-input so custom IDs are typeable', () => {
    for (const key of [
      'tts.openai.voice',
      'tts.openai.model',
      'tts.elevenlabs.voice_id',
      'tts.edge.voice',
      'tts.xai.voice_id',
      'tts.piper.voice'
    ]) {
      expect(FREE_INPUT_KEYS.has(key), key).toBe(true)
    }
  })

  it('keeps closed enums (devices, providers) out of the free-input set', () => {
    expect(FREE_INPUT_KEYS.has('tts.provider')).toBe(false)
    expect(FREE_INPUT_KEYS.has('tts.neutts.device')).toBe(false)
    expect(FREE_INPUT_KEYS.has('stt.provider')).toBe(false)
  })

  it('every free-input voice key that lives in the Voice section has suggestions or is intentionally bare', () => {
    // Free-input keys don't *require* ENUM_OPTIONS (an empty datalist is
    // fine), but any that do declare options must be actual Voice-section
    // fields — a typo'd key here would silently do nothing.
    for (const key of FREE_INPUT_KEYS) {
      expect(voiceKeys, key).toContain(key)
    }
  })
})
