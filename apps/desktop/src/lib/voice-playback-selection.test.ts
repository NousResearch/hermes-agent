import { beforeEach, describe, expect, it, vi } from 'vitest'

const { resolveGatewayWsUrl, speakText } = vi.hoisted(() => ({
  resolveGatewayWsUrl: vi.fn(),
  speakText: vi.fn()
}))

vi.mock('@hermes/shared', () => ({ resolveGatewayWsUrl }))
vi.mock('@/hermes', () => ({ getApiRequestProfile: () => null, speakText }))

import { $voicePlayback } from '@/store/voice-playback'

import {
  getVoicePlaybackSequence,
  isVoicePlaybackSequenceCurrent,
  playSelectedSpeechText,
  startSpeechStream,
  stopVoicePlayback
} from './voice-playback'

class FakeWebSocket {
  static readonly CLOSED = 3
  static readonly CONNECTING = 0
  static readonly OPEN = 1
  static instances: FakeWebSocket[] = []

  binaryType = ''
  onclose: null | (() => void) = null
  onerror: null | (() => void) = null
  onmessage: null | ((event: { data: unknown }) => void) = null
  onopen: null | (() => void) = null
  readyState = FakeWebSocket.CONNECTING

  constructor() {
    FakeWebSocket.instances.push(this)
  }

  close() {
    this.readyState = FakeWebSocket.CLOSED
  }

  send() {}
}

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
    FakeWebSocket.instances = []
    resolveGatewayWsUrl.mockReset()
    speakText.mockReset()
    speakText.mockResolvedValue({ data_url: 'data:audio/mpeg;base64,voice' })
    vi.stubGlobal('Audio', FakeAudio)
    vi.stubGlobal('WebSocket', FakeWebSocket)
    Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: {} })
  })

  it('uses the normal Hermes voice pipeline for only the selected text', async () => {
    const playback = playSelectedSpeechText('  selected words only  ')

    expect($voicePlayback.get()).toMatchObject({
      messageId: 'selection-read-aloud',
      source: 'read-aloud',
      status: 'preparing'
    })
    await vi.waitFor(() => expect(speakText).toHaveBeenCalledOnce())
    expect(speakText).toHaveBeenCalledWith('selected words only')

    await expect(playback).resolves.toBe(true)
    expect($voicePlayback.get().status).toBe('idle')
  })

  it('invalidates a captured ownership sequence when newer playback starts', async () => {
    let resolveSpeech!: (value: { data_url: string }) => void

    resolveGatewayWsUrl.mockRejectedValueOnce(new Error('stream unavailable'))
    speakText.mockImplementationOnce(() => new Promise(resolve => (resolveSpeech = resolve)))

    const capturedSequence = getVoicePlaybackSequence()
    const playback = playSelectedSpeechText('newer selected text')

    await vi.waitFor(() => expect(speakText).toHaveBeenCalledOnce())
    expect(isVoicePlaybackSequenceCurrent(capturedSequence)).toBe(false)

    stopVoicePlayback()
    resolveSpeech({ data_url: 'data:audio/mpeg;base64,voice' })
    await expect(playback).resolves.toBe(false)
  })

  it('plays every chunk of a long selection through the Hermes voice in order', async () => {
    const text = Array.from(
      { length: 140 },
      (_, index) => `Sentence ${index + 1} explains another part of the recommendation clearly.`
    ).join(' ')

    await expect(playSelectedSpeechText(text)).resolves.toBe(true)

    const spokenChunks = speakText.mock.calls.map(([chunk]) => chunk as string)

    expect(spokenChunks.length).toBeGreaterThan(1)
    expect(spokenChunks.every(chunk => chunk.length < 2_000)).toBe(true)
    expect(spokenChunks.join(' ')).toBe(text)
    expect($voicePlayback.get().status).toBe('idle')
  })

  it('does not request a later fallback chunk after playback is stopped', async () => {
    let resolveFirst!: (value: { data_url: string }) => void
    speakText.mockImplementationOnce(
      () =>
        new Promise(resolve => {
          resolveFirst = resolve
        })
    )

    const text = Array.from(
      { length: 140 },
      (_, index) => `Sentence ${index + 1} explains another part of the recommendation clearly.`
    ).join(' ')

    const playback = playSelectedSpeechText(text)

    await vi.waitFor(() => expect(speakText).toHaveBeenCalledOnce())
    stopVoicePlayback()
    resolveFirst({ data_url: 'data:audio/mpeg;base64,voice' })

    await expect(playback).resolves.toBe(false)
    expect(speakText).toHaveBeenCalledOnce()
    expect($voicePlayback.get().status).toBe('idle')
  })

  it('honors a lower configured provider cap from the WebSocket fallback frame', async () => {
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { getConnection: vi.fn(async () => ({})) }
    })
    resolveGatewayWsUrl.mockResolvedValue('ws://localhost/api/ws')

    const text = Array.from(
      { length: 90 },
      (_, index) => `Sentence ${index + 1} carries every selected word through the configured provider.`
    ).join(' ')

    const playback = playSelectedSpeechText(text)
    await vi.waitFor(() => expect(FakeWebSocket.instances).toHaveLength(1))
    FakeWebSocket.instances[0].onmessage?.({
      data: JSON.stringify({ max_text_length: 1000, type: 'fallback' })
    })

    await expect(playback).resolves.toBe(true)
    const spokenChunks = speakText.mock.calls.map(([chunk]) => chunk as string)

    expect(spokenChunks.length).toBeGreaterThan(1)
    expect(spokenChunks.every(chunk => chunk.length <= 1000)).toBe(true)
    expect(spokenChunks.join(' ')).toBe(text)
  })

  it('does not let a pending live stream replace newer selection playback', async () => {
    let resolveOldUrl!: (url: string) => void
    let resolveSpeech!: (value: { data_url: string }) => void

    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { getConnection: vi.fn(async () => ({})) }
    })
    resolveGatewayWsUrl
      .mockImplementationOnce(() => new Promise<string>(resolve => (resolveOldUrl = resolve)))
      .mockRejectedValueOnce(new Error('stream unavailable'))
    speakText.mockImplementationOnce(() => new Promise(resolve => (resolveSpeech = resolve)))

    const olderStart = startSpeechStream({ source: 'voice-conversation' })
    await vi.waitFor(() => expect(resolveGatewayWsUrl).toHaveBeenCalledOnce())

    const selectionPlayback = playSelectedSpeechText('newer selected text')
    await vi.waitFor(() => expect(speakText).toHaveBeenCalledOnce())
    expect($voicePlayback.get()).toMatchObject({
      messageId: 'selection-read-aloud',
      status: 'preparing'
    })

    resolveOldUrl('ws://localhost/api/ws')
    await expect(olderStart).resolves.toBeUndefined()
    expect($voicePlayback.get()).toMatchObject({
      messageId: 'selection-read-aloud',
      status: 'preparing'
    })

    stopVoicePlayback()
    resolveSpeech({ data_url: 'data:audio/mpeg;base64,voice' })
    await expect(selectionPlayback).resolves.toBe(false)
  })

  it('does not publish idle from an old live session after selection playback owns the state', async () => {
    let resolveSelectionUrl!: (url: string) => void

    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { getConnection: vi.fn(async () => ({})) }
    })
    resolveGatewayWsUrl
      .mockResolvedValueOnce('ws://localhost/api/ws')
      .mockImplementationOnce(() => new Promise<string>(resolve => (resolveSelectionUrl = resolve)))

    const oldSession = await startSpeechStream({ source: 'voice-conversation' })
    expect(oldSession).toBeTruthy()

    const selectionPlayback = playSelectedSpeechText('newer selected text')
    await vi.waitFor(() => expect(resolveGatewayWsUrl).toHaveBeenCalledTimes(2))
    await Promise.resolve()

    expect($voicePlayback.get()).toMatchObject({
      messageId: 'selection-read-aloud',
      status: 'preparing'
    })

    stopVoicePlayback()
    resolveSelectionUrl('ws://localhost/api/ws')
    await expect(selectionPlayback).resolves.toBe(false)
  })

  it('does not revive a pending live stream after explicit cancellation', async () => {
    let resolveUrl!: (url: string) => void

    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { getConnection: vi.fn(async () => ({})) }
    })
    resolveGatewayWsUrl.mockImplementationOnce(() => new Promise<string>(resolve => (resolveUrl = resolve)))

    const pending = startSpeechStream({ source: 'voice-conversation' })
    await vi.waitFor(() => expect(resolveGatewayWsUrl).toHaveBeenCalledOnce())
    stopVoicePlayback()
    resolveUrl('ws://localhost/api/ws')

    await expect(pending).resolves.toBeUndefined()
    expect($voicePlayback.get().status).toBe('idle')
  })
})
