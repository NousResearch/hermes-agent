import { describe, expect, it } from 'vitest'

import type { HermesConfigRecord } from '@/types/hermes'

import { voiceFieldVisible } from './helpers'

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

  it('tracks a provider switch', () => {
    expect(voiceFieldVisible('tts.openai.voice', cfg({ tts: { provider: 'openai', openai: {} } }))).toBe(true)
    expect(voiceFieldVisible('tts.edge.voice', cfg({ tts: { provider: 'openai', openai: {} } }))).toBe(false)
  })

  it('shows the OpenRouter STT options only under OpenRouter', () => {
    const openrouter = cfg({ stt: { enabled: true, provider: 'openrouter', openrouter: {} } })
    expect(voiceFieldVisible('stt.openrouter.model', openrouter)).toBe(true)
    expect(voiceFieldVisible('stt.openai.model', openrouter)).toBe(false)
    // ...and not while another STT provider is selected.
    expect(voiceFieldVisible('stt.openrouter.model', cfg())).toBe(false)
  })

  it('shows a command-declared provider\'s fields while it is selected', () => {
    // Command providers are keyed one level deeper (`stt.providers.<name>.model`); the old
    // regex read the provider as "providers", so these fields never rendered at all.
    const selected = cfg({
      stt: { enabled: true, provider: 'openrouter', providers: { openrouter: { type: 'command', command: 'run' } } }
    })
    expect(voiceFieldVisible('stt.providers.openrouter.model', selected)).toBe(true)

    const otherSelected = cfg({
      stt: { enabled: true, provider: 'groq', providers: { openrouter: { type: 'command', command: 'run' } } }
    })
    expect(voiceFieldVisible('stt.providers.openrouter.model', otherSelected)).toBe(false)

    const ttsSide = cfg({ tts: { provider: 'mybackend', providers: { mybackend: { type: 'command', command: 'run' } } } })
    expect(voiceFieldVisible('tts.providers.mybackend.voice', ttsSide)).toBe(true)
    expect(voiceFieldVisible('tts.providers.mybackend.voice', cfg())).toBe(false)
  })
})
