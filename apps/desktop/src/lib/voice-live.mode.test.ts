import { describe, expect, it } from 'vitest'

import {
  gptLiveEngineAvailable,
  grokLiveEngineAvailable,
  grokLiveEngineReason,
  isVoiceChatMode,
  normalizeVoiceChatMode,
  type VoiceLiveStatus
} from './voice-live'

function status(patch: Partial<VoiceLiveStatus> = {}): VoiceLiveStatus {
  return {
    available: false,
    mode: 'chained',
    model: 'gpt-live-1',
    reason: 'no OpenAI API key (set OPENAI_API_KEY or voice.gpt_live.api_key)',
    voice: 'marin',
    ...patch
  }
}

describe('voice chat mode', () => {
  it('accepts grok-live beside chained and gpt-live', () => {
    expect(normalizeVoiceChatMode('grok-live')).toBe('grok-live')
    expect(normalizeVoiceChatMode('gpt-live')).toBe('gpt-live')
    expect(normalizeVoiceChatMode('chained')).toBe('chained')
    expect(normalizeVoiceChatMode('nope')).toBe('chained')
    expect(isVoiceChatMode('grok-live')).toBe(true)
    expect(isVoiceChatMode('realtime')).toBe(false)
  })

  it('keeps each engine available from its own credential, not the saved mode', () => {
    const both = status({
      auth: 'oauth',
      available: false,
      gpt_live_available: true,
      gpt_live_reason: null,
      grok_auth: 'oauth',
      grok_available: true,
      grok_reason: null,
      mode: 'grok-live',
      model: 'grok-voice-latest',
      reason: null,
      voice: 'eve'
    })

    expect(grokLiveEngineAvailable(both)).toBe(true)
    expect(gptLiveEngineAvailable(both)).toBe(true)

    const neither = status({
      available: false,
      gpt_live_available: false,
      grok_available: false,
      grok_reason: 'no xAI credential (hermes auth add xai-oauth or XAI_API_KEY)',
      mode: 'grok-live'
    })

    expect(grokLiveEngineAvailable(neither)).toBe(false)
    expect(grokLiveEngineReason(neither)).toContain('XAI_API_KEY')
    expect(gptLiveEngineAvailable(neither)).toBe(false)
  })

  it('falls back when an older status has no per-engine fields', () => {
    expect(gptLiveEngineAvailable(status({ available: true, mode: 'gpt-live', reason: null }))).toBe(true)
    expect(grokLiveEngineAvailable(status({ available: true, mode: 'gpt-live', reason: null }))).toBe(false)
    expect(grokLiveEngineAvailable(status({ available: true, mode: 'grok-live', reason: null }))).toBe(true)
    expect(gptLiveEngineAvailable(status({ available: true, mode: 'grok-live', reason: null }))).toBe(false)
  })
})
