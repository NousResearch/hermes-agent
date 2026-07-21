import { beforeEach, describe, expect, it, vi } from 'vitest'

const { speakText } = vi.hoisted(() => ({ speakText: vi.fn() }))

vi.mock('@/hermes', () => ({ speakText }))

import { $voicePlayback } from '@/store/voice-playback'

import { playSelectedSpeechText, stopVoicePlayback } from './voice-playback'

class FakeAudio {
  src: string
  private listeners = new Map<string, Set<() => void>>()

  constructor(src: string) {
    this.src = src
  }

  addEventListener(name: string, listener: () => void) {
    const listeners = this.listeners.get(name) ?? new Set()

    listeners.add(listener)
    this.listeners.set(name, listeners)
  }

  removeEventListener(name: string, listener: () => void) {
    this.listeners.get(name)?.delete(listener)
  }

  load() {}
  pause() {}

  async play() {
    queueMicrotask(() => this.listeners.get('ended')?.forEach(listener => listener()))
  }
}

describe('playSelectedSpeechText', () => {
  beforeEach(() => {
    stopVoicePlayback()
    speakText.mockReset()
    speakText.mockResolvedValue({ data_url: 'data:audio/mpeg;base64,voice' })
    vi.stubGlobal('Audio', FakeAudio)
  })

  it('uses the normal Hermes voice pipeline for only the selected text', async () => {
    const playback = playSelectedSpeechText('  selected words only  ')

    expect(speakText).toHaveBeenCalledOnce()
    expect(speakText).toHaveBeenCalledWith('selected words only')
    expect($voicePlayback.get()).toMatchObject({
      messageId: 'selection-read-aloud',
      source: 'read-aloud',
      status: 'preparing'
    })

    await expect(playback).resolves.toBe(true)
    expect($voicePlayback.get().status).toBe('idle')
  })

  it('plays every chunk of a long selection through the Hermes voice in order', async () => {
    const text = Array.from(
      { length: 140 },
      (_, index) => `Sentence ${index + 1} explains another part of the recommendation clearly.`
    ).join(' ')

    await expect(playSelectedSpeechText(text)).resolves.toBe(true)

    const spokenChunks = speakText.mock.calls.map(([chunk]) => chunk as string)

    expect(spokenChunks.length).toBeGreaterThan(1)
    expect(spokenChunks.every(chunk => chunk.length <= 4_500)).toBe(true)
    expect(spokenChunks.join(' ')).toBe(text)
    expect($voicePlayback.get().status).toBe('idle')
  })
})
