import { describe, expect, it } from 'vitest'

import {
  createMiniMaxCloneVoiceId,
  miniMaxVoiceErrorMessage,
  miniMaxVoiceName,
  validateMiniMaxCloneInput
} from './minimax-voice-workflows'

describe('MiniMax voice workflow helpers', () => {
  it('creates a unique clone id that cannot collide with a system id', () => {
    const id = createMiniMaxCloneVoiceId('Hermes', new Date('2026-07-10T12:34:56Z'))

    expect(id).toBe('HermesClone0710123456')
    expect(id).not.toBe('Korean_GentleBoss')
  })

  it('formats duplicate ids as an existing-voice action', () => {
    expect(miniMaxVoiceErrorMessage('voice clone voice id duplicate')).toContain('已有音色')
  })

  it('encodes an existing MiniMax ID for MoneyPrinter video generation', () => {
    expect(miniMaxVoiceName('Korean_GentleBoss')).toBe('minimax:Korean_GentleBoss')
  })

  it('requires prompt audio and prompt text together', () => {
    expect(
      validateMiniMaxCloneInput({
        cloneFile: true,
        promptFile: true,
        promptText: '',
        voiceId: 'HermesClone001'
      })
    ).toBe('参考音频和参考音频文本必须同时提供。')
  })

  it('allows a new id and source audio without paid activation', () => {
    expect(
      validateMiniMaxCloneInput({
        cloneFile: true,
        promptFile: false,
        promptText: '',
        voiceId: 'HermesClone001'
      })
    ).toBe('')
  })
})
