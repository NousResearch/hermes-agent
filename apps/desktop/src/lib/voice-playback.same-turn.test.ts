import { afterEach, describe, expect, it, vi } from 'vitest'

import { $voicePlayback } from '@/store/voice-playback'

import { playSpeechText, stopVoicePlayback } from './voice-playback'

vi.mock('@/lib/voice-client-direct', () => ({
  cutSentences: (text: string) => [text],
  directTtsConfig: vi.fn(async () => null),
  synthesizeSpeechClientDirect: vi.fn()
}))

vi.mock('@/hermes', () => ({
  getApiRequestConnection: () => null,
  getApiRequestProfile: () => null,
  speakText: vi.fn(async () => {
    throw new Error('no audio in test')
  })
}))

describe('playSpeechText same-turn start', () => {
  afterEach(() => {
    stopVoicePlayback()
  })

  it('does not stop an in-flight play when a second start is the same turn', async () => {
    const first = playSpeechText('hello there', {
      messageId: 'assistant-stream-1',
      source: 'read-aloud',
      turnKey: 'session:0'
    })

    const sequenceAfterFirst = $voicePlayback.get().sequence

    const second = playSpeechText('hello there', {
      messageId: 'durable-42',
      source: 'read-aloud',
      turnKey: 'session:0'
    })

    expect($voicePlayback.get().sequence).toBe(sequenceAfterFirst)
    expect($voicePlayback.get().status).not.toBe('idle')

    stopVoicePlayback()
    await Promise.allSettled([first, second])
  })

  it('still replaces playback when the next start is a different turn', async () => {
    const first = playSpeechText('first reply', {
      messageId: 'a1',
      source: 'read-aloud',
      turnKey: 'session:0'
    })

    const sequenceAfterFirst = $voicePlayback.get().sequence

    const second = playSpeechText('second reply', {
      messageId: 'a2',
      source: 'read-aloud',
      turnKey: 'session:1'
    })

    expect($voicePlayback.get().sequence).toBeGreaterThan(sequenceAfterFirst)

    stopVoicePlayback()
    await Promise.allSettled([first, second])
  })

  it('does not let the idle notification from its own stop start a second clip of the same turn', async () => {
    const stops: number[] = []

    const unlisten = $voicePlayback.listen(state => {
      if (state.status !== 'idle') {
        return
      }

      stops.push(state.sequence)
      void playSpeechText('hello there', {
        messageId: 'durable-42',
        source: 'read-aloud',
        turnKey: 'session:0'
      })
    })

    const before = $voicePlayback.get().sequence

    const first = playSpeechText('hello there', {
      messageId: 'assistant-stream-1',
      source: 'read-aloud',
      turnKey: 'session:0'
    })

    unlisten()
    // One stop: the start itself. A second playSpeechText from the idle edge would bump sequence again.
    expect($voicePlayback.get().sequence).toBe(before + 1)
    expect(stops.length).toBeGreaterThan(0)

    stopVoicePlayback()
    await Promise.allSettled([first])
  })
})
