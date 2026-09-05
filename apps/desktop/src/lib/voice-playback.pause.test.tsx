import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { VoicePlaybackActivity } from '@/app/chat/composer/voice-activity'
import { $voicePlayback } from '@/store/voice-playback'

import { playSpeechText, stopVoicePlayback, toggleVoicePlaybackPaused } from './voice-playback'

const mocks = vi.hoisted(() => ({ direct: vi.fn(), speak: vi.fn() }))
vi.mock('@/hermes', () => ({
  getApiRequestConnection: () => null,
  getApiRequestProfile: () => null,
  speakText: (...args: unknown[]) => mocks.speak(...args)
}))
vi.mock('@/lib/voice-client-direct', () => ({
  directTtsConfig: () => mocks.direct(),
  synthesizeSpeechClientDirect: async () => new ArrayBuffer(4),
  cutSentences: (text: string) => ({ rest: '', sentences: text ? [text] : [] })
}))

class AudioFixture extends EventTarget {
  static instances: AudioFixture[] = []
  src: string
  paused = true
  currentTime = 7
  constructor(src: string) {
    super()
    this.src = src
    AudioFixture.instances.push(this)
  }
  play = vi.fn(async () => {
    this.paused = false
  })
  pause = vi.fn(() => {
    this.paused = true
  })
  load = vi.fn()
}

class ContextFixture {
  static instances: ContextFixture[] = []
  state = 'running'
  currentTime = 0
  destination = {}
  constructor() {
    ContextFixture.instances.push(this)
  }
  suspend = vi.fn(async () => {
    this.state = 'suspended'
  })
  resume = vi.fn(async () => {
    this.state = 'running'
  })
  close = vi.fn(async () => {
    this.state = 'closed'
  })
  createBuffer = (_channels: number, length: number, rate: number) => ({
    duration: length / rate,
    getChannelData: () => new Float32Array(length)
  })
  createMediaElementSource = () => ({ connect: vi.fn() })
  createAnalyser = () => ({ connect: vi.fn(), frequencyBinCount: 256, getByteFrequencyData: vi.fn() })
  createBufferSource = () => ({ connect: vi.fn(), start: vi.fn(), buffer: null })
}

class SocketFixture {
  static OPEN = 1
  static CONNECTING = 0
  static instances: SocketFixture[] = []
  readyState = 1
  binaryType = ''
  onmessage: ((event: { data: string | ArrayBuffer }) => void) | null = null
  onclose: (() => void) | null = null
  constructor() {
    SocketFixture.instances.push(this)
  }
  send = vi.fn()
  close = vi.fn()
}

beforeEach(() => {
  vi.useFakeTimers()
  AudioFixture.instances = []
  ContextFixture.instances = []
  SocketFixture.instances = []
  vi.stubGlobal('Audio', AudioFixture)
  vi.stubGlobal('AudioContext', ContextFixture)
  vi.stubGlobal('WebSocket', SocketFixture)
  vi.spyOn(URL, 'createObjectURL').mockReturnValue('blob:audio')
  vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => undefined)
  mocks.direct.mockResolvedValue(null)
  mocks.speak.mockResolvedValue({ data_url: 'data:audio/mp3;base64,AAAA' })
  Reflect.deleteProperty(window, 'hermesDesktop')
})
afterEach(() => {
  stopVoicePlayback()
  cleanup()
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
  vi.useRealTimers()
  Reflect.deleteProperty(window, 'hermesDesktop')
})

it.each(['direct', 'relay', 'fallback'])(
  'pauses %s playback without completing or losing position',
  async transport => {
    if (transport === 'direct') {mocks.direct.mockResolvedValue({ provider: 'test' })}

    if (transport === 'relay')
      {Object.defineProperty(window, 'hermesDesktop', {
        configurable: true,
        value: {
          getConnection: async () => ({
            authMode: 'token',
            baseUrl: 'http://localhost',
            wsUrl: 'ws://localhost/api/ws'
          }),
          getGatewayWsUrl: async () => ({ ok: true, wsUrl: 'ws://localhost/api/ws?token=test' })
        }
      })}

    let completed = false

    const playing = playSpeechText('Read this.', { source: 'read-aloud', messageId: 'message-1' }).then(value => {
      completed = true

      return value
    })

    await vi.advanceTimersByTimeAsync(0)

    if (transport === 'relay') {
      const ws = SocketFixture.instances[0]!
      ws.onmessage!({ data: JSON.stringify({ type: 'start', sample_rate: 24000 }) })
      ws.onmessage!({ data: new ArrayBuffer(48000) })
      ws.onmessage!({ data: JSON.stringify({ type: 'end' }) })
    }

    expect($voicePlayback.get().status).toBe('speaking')
    const originalSequence = $voicePlayback.get().sequence
    await toggleVoicePlaybackPaused()
    expect($voicePlayback.get().status).toBe('paused')
    await vi.advanceTimersByTimeAsync(30000)
    expect(completed).toBe(false)
    expect($voicePlayback.get().sequence).toBe(originalSequence)

    if (transport === 'relay') {expect(ContextFixture.instances[0]!.state).toBe('suspended')}
    else {
      expect(AudioFixture.instances[0]!.paused).toBe(true)
      expect(AudioFixture.instances[0]!.currentTime).toBe(7)
    }

    await toggleVoicePlaybackPaused()
    expect($voicePlayback.get().status).toBe('speaking')

    if (transport === 'relay') {
      ContextFixture.instances[0]!.currentTime = 2
      await vi.advanceTimersByTimeAsync(1500)
    } else {AudioFixture.instances[0]!.dispatchEvent(new Event('ended'))}

    expect(await playing).toBe(true)
    expect($voicePlayback.get().status).toBe('idle')
  }
)

it('offers pause/resume and can stop while paused', async () => {
  const playing = playSpeechText('Read this.', { source: 'read-aloud' })
  await vi.advanceTimersByTimeAsync(0)
  render(<VoicePlaybackActivity />)
  await act(async () => fireEvent.click(screen.getByRole('button', { name: 'Pause playback' })))
  expect(screen.getByText('Playback paused')).toBeTruthy()
  expect(screen.getByRole('button', { name: 'Resume playback' })).toBeTruthy()
  await act(async () => fireEvent.click(screen.getByRole('button', { name: 'Stop playback' })))
  expect(await playing).toBe(false)
  expect($voicePlayback.get().status).toBe('idle')
  expect(screen.queryByRole('status')).toBeNull()
})
