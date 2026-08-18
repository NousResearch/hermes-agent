import { describe, expect, it } from 'vitest'

import type { HermesConfigRecord } from '@/types/hermes'

import { voiceFieldVisible } from './config-settings'

const cfg = (over: Record<string, unknown> = {}): HermesConfigRecord =>
  ({
    tts: { provider: 'edge', edge: {}, openai: {} },
    stt: { enabled: true, provider: 'local', local: {}, groq: {} },
    ...over
  }) as unknown as HermesConfigRecord

describe('voiceFieldVisible', () => {
  it('always shows top-level + non-provider keys', () => {
    const config = cfg()

    for (const key of ['tts.provider', 'stt.enabled', 'stt.provider', 'voice.auto_tts', 'voice.record_key']) {
      expect(voiceFieldVisible(key, config)).toBe(true)
    }
  })

  it('shows only the selected TTS provider sub-fields', () => {
    const config = cfg()
    expect(voiceFieldVisible('tts.edge.voice', config)).toBe(true)
    expect(voiceFieldVisible('tts.openai.voice', config)).toBe(false)
    expect(voiceFieldVisible('tts.elevenlabs.voice_id', config)).toBe(false)
  })

  it('shows only the selected STT provider sub-fields', () => {
    const config = cfg()
    expect(voiceFieldVisible('stt.local.model', config)).toBe(true)
    expect(voiceFieldVisible('stt.groq.model', config)).toBe(false)
  })

  it('hides every STT provider sub-field when STT is disabled', () => {
    const config = cfg({ stt: { enabled: false, provider: 'local', local: {} } })
    expect(voiceFieldVisible('stt.local.model', config)).toBe(false)
    // ...but the enable/provider toggles themselves stay visible.
    expect(voiceFieldVisible('stt.enabled', config)).toBe(true)
    expect(voiceFieldVisible('stt.provider', config)).toBe(true)
  })

  it('shows the dictation route always, and its loopback knobs only in local mode', () => {
    const loopbackKeys = ['voice.dictation.local_stt_port', 'voice.dictation.local_stt_model']

    // Unset config = the backend default, so the loopback knobs stay hidden.
    expect(voiceFieldVisible('voice.dictation.stt', cfg())).toBe(true)

    for (const key of loopbackKeys) {
      expect(voiceFieldVisible(key, cfg()), key).toBe(false)
    }

    const local = cfg({ voice: { dictation: { stt: 'local' } } })
    expect(voiceFieldVisible('voice.dictation.stt', local)).toBe(true)

    for (const key of loopbackKeys) {
      expect(voiceFieldVisible(key, local), key).toBe(true)
    }

    // STT being off is about backend providers; it must not hide the route.
    const sttOff = cfg({ stt: { enabled: false }, voice: { dictation: { stt: 'local' } } })

    for (const key of loopbackKeys) {
      expect(voiceFieldVisible(key, sttOff), key).toBe(true)
    }
  })

  it('tracks a provider switch', () => {
    expect(voiceFieldVisible('tts.openai.voice', cfg({ tts: { provider: 'openai', openai: {} } }))).toBe(true)
    expect(voiceFieldVisible('tts.edge.voice', cfg({ tts: { provider: 'openai', openai: {} } }))).toBe(false)
  })
})
