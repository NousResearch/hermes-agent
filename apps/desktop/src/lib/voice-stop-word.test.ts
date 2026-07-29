import { describe, expect, it } from 'vitest'

import { isVoiceStopCommand } from './voice-stop-word'

describe('isVoiceStopCommand', () => {
  it('matches bare stop commands', () => {
    for (const phrase of ['stop', 'Stop', 'STOP', 'stop.', 'stop!', ' stop ', 'stop…']) {
      expect(isVoiceStopCommand(phrase)).toBe(true)
    }
  })

  it('matches multi-word stop phrases', () => {
    for (const phrase of [
      'stop listening',
      'stop it',
      'please stop',
      'stop please',
      "that's all",
      'that is all',
      'never mind',
      'nevermind',
      'end conversation',
      'end the conversation',
      'goodbye',
      'bye',
      'cancel'
    ]) {
      expect(isVoiceStopCommand(phrase)).toBe(true)
    }
  })

  it('matches stop commands addressed to Hermes', () => {
    for (const phrase of ['hermes stop', 'hey hermes stop', 'hey hermes, stop', 'ok stop', 'okay stop']) {
      expect(isVoiceStopCommand(phrase)).toBe(true)
    }
  })

  it('does NOT match substantive requests that merely contain "stop"', () => {
    for (const phrase of [
      'stop the docker container',
      'how do I stop a running process',
      'can you stop the deployment',
      'stop the music and play something else',
      "don't stop now",
      'the bus stop is closed'
    ]) {
      expect(isVoiceStopCommand(phrase)).toBe(false)
    }
  })

  it('does not match bare address words or empty input', () => {
    for (const phrase of ['', '  ', 'hermes', 'hey hermes', 'ok', 'okay', 'hey']) {
      expect(isVoiceStopCommand(phrase)).toBe(false)
    }
  })

  it('does not match unrelated short utterances', () => {
    for (const phrase of ['hello', 'yes', 'what time is it', 'thanks']) {
      expect(isVoiceStopCommand(phrase)).toBe(false)
    }
  })
})
