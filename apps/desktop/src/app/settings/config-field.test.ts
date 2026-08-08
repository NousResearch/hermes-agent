import { describe, expect, it } from 'vitest'

import { SCHEMA_OPTION_LABELS } from './constants'

describe('SCHEMA_OPTION_LABELS', () => {
  it('uses the intended casing only for matching config keys', () => {
    expect(SCHEMA_OPTION_LABELS['terminal.backend']?.ssh).toBe('SSH')
    expect(SCHEMA_OPTION_LABELS['tts.provider']).toMatchObject({
      openai: 'OpenAI',
      xai: 'xAI',
      elevenlabs: 'ElevenLabs',
      minimax: 'MiniMax',
      neutts: 'NeuTTS',
      kittentts: 'KittenTTS'
    })
    expect(SCHEMA_OPTION_LABELS['stt.provider']).toMatchObject({
      openai: 'OpenAI',
      xai: 'xAI',
      elevenlabs: 'ElevenLabs'
    })
    expect(SCHEMA_OPTION_LABELS['tts.neutts.device']).toMatchObject({ cpu: 'CPU', cuda: 'CUDA', mps: 'MPS' })
    expect(SCHEMA_OPTION_LABELS['unrelated.setting']).toBeUndefined()
  })
})
