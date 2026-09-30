import { describe, expect, it } from 'vitest'

import { interceptsTypedVoiceStop, isVoiceStopCommand } from './voice-stop-word'

describe('isVoiceStopCommand', () => {
  it('matches bare stop commands', () => {
    for (const phrase of ['stop', 'Stop', 'STOP', 'stop.', 'stop!', ' stop ', 'stop…']) {
      expect(isVoiceStopCommand(phrase)).toBe(true)
    }
  })

  it('matches explicitly configured multi-word stop phrases', () => {
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
      expect(isVoiceStopCommand(phrase, [phrase])).toBe(true)
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

  it('uses only configured phrases and compares whole Unicode-equivalent utterances', () => {
    const phrases = ['그만', '대화 종료', 'arrêt']

    for (const transcript of ['그만'.normalize('NFD'), '“대화 종료”!', 'ARRÊT'.normalize('NFD')]) {
      expect(isVoiceStopCommand(transcript, phrases)).toBe(true)
    }

    for (const transcript of ['그만하고 다음 작업', 'stop', 'never mind', 'cancel']) {
      expect(isVoiceStopCommand(transcript, phrases)).toBe(false)
    }

    expect(isVoiceStopCommand('stop', [])).toBe(false)
    expect(isVoiceStopCommand('never mind')).toBe(false)
    expect(isVoiceStopCommand('stop the docker container')).toBe(false)
  })
})

describe('interceptsTypedVoiceStop', () => {
  it('intercepts a typed bare stop command while the conversation is active', () => {
    for (const text of ['stop', 'Stop.', 'hey hermes, stop']) {
      expect(interceptsTypedVoiceStop(true, text)).toBe(true)
    }
  })

  it('never intercepts when the voice conversation is inactive', () => {
    for (const text of ['stop', 'never mind', 'goodbye']) {
      expect(interceptsTypedVoiceStop(false, text)).toBe(false)
    }
  })

  it('passes through substantive messages during a conversation', () => {
    for (const text of ['stop the docker container', 'how do I stop a process', 'hello']) {
      expect(interceptsTypedVoiceStop(true, text)).toBe(false)
    }
  })

  it('passes through when attachments ride along (real payload)', () => {
    expect(interceptsTypedVoiceStop(true, 'stop', 1)).toBe(false)
  })

  it('uses the same configured list for typed and spoken stops while preserving attachment payloads', () => {
    for (const phrases of [[], ['그만']] as const) {
      for (const text of ['stop', '그만'.normalize('NFD'), '그만하고 다음 작업']) {
        expect(interceptsTypedVoiceStop(true, text, 0, phrases)).toBe(isVoiceStopCommand(text, phrases))
        expect(interceptsTypedVoiceStop(false, text, 0, phrases)).toBe(false)
        expect(interceptsTypedVoiceStop(true, text, 1, phrases)).toBe(false)
      }
    }
  })
})
