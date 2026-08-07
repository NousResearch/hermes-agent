import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { startSpeechStream, stopVoicePlayback } from './voice-playback'

const mocks = vi.hoisted(() => ({
  getApiRequestProfile: vi.fn(() => null),
  resolveGatewayWsUrl: vi.fn(async () => 'ws://localhost/api/ws?token=test'),
  speakText: vi.fn()
}))

vi.mock('@hermes/shared', () => ({
  resolveGatewayWsUrl: mocks.resolveGatewayWsUrl
}))

vi.mock('@/hermes', () => ({
  getApiRequestProfile: mocks.getApiRequestProfile,
  speakText: mocks.speakText
}))

class FakeWebSocket {
  static readonly CLOSED = 3
  static readonly CONNECTING = 0
  static readonly OPEN = 1
  static instances: FakeWebSocket[] = []

  binaryType = ''
  onclose: null | (() => void) = null
  onerror: null | (() => void) = null
  onmessage: null | ((event: MessageEvent) => void) = null
  onopen: null | (() => void) = null
  readyState = FakeWebSocket.CONNECTING
  readonly sent: string[] = []

  constructor(readonly url: string) {
    FakeWebSocket.instances.push(this)
  }

  close() {
    this.readyState = FakeWebSocket.CLOSED
  }

  emit(data: ArrayBuffer | string) {
    this.onmessage?.({ data } as MessageEvent)
  }

  open() {
    this.readyState = FakeWebSocket.OPEN
    this.onopen?.()
  }

  send(data: string) {
    this.sent.push(data)
  }
}

class FakeAudioContext {
  static instances: FakeAudioContext[] = []

  readonly close = vi.fn(async () => undefined)
  readonly createBufferSource = vi.fn(() => ({
    buffer: null,
    connect: vi.fn(),
    start: vi.fn()
  }))
  currentTime = 0
  readonly destination = {}
  readonly resume = vi.fn(async () => undefined)
  state: AudioContextState = 'running'

  constructor() {
    FakeAudioContext.instances.push(this)
  }

  async decodeAudioData(data: ArrayBuffer): Promise<AudioBuffer> {
    const marker = new Uint8Array(data)[0]

    if (marker === 2) {
      throw new Error('invalid encoded clip')
    }

    return { duration: 0.01 } as AudioBuffer
  }
}

async function flushPromises() {
  await Promise.resolve()
  await Promise.resolve()
  await Promise.resolve()
  await Promise.resolve()
}

describe('encoded voice playback', () => {
  beforeEach(() => {
    FakeWebSocket.instances = []
    FakeAudioContext.instances = []
    vi.stubGlobal('WebSocket', FakeWebSocket)
    vi.stubGlobal('AudioContext', FakeAudioContext)
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { getConnection: vi.fn(async () => ({ wsUrl: 'ws://localhost/api/ws?token=test' })) }
    })
  })

  afterEach(() => {
    stopVoicePlayback()
    vi.unstubAllGlobals()
    Reflect.deleteProperty(window, 'hermesDesktop')
    vi.clearAllMocks()
  })

  it('stops decoding after an invalid middle clip and drains scheduled audio', async () => {
    const session = await startSpeechStream({ source: 'voice-conversation' })
    expect(session).not.toBeNull()

    const socket = FakeWebSocket.instances[0]
    expect(new URL(socket.url).searchParams.get('audio_protocol')).toBe('2')
    socket.open()
    socket.emit(JSON.stringify({ type: 'start', encoding: 'encoded' }))
    socket.emit(Uint8Array.of(1).buffer)
    socket.emit(Uint8Array.of(2).buffer)
    socket.emit(Uint8Array.of(3).buffer)
    socket.emit(JSON.stringify({ type: 'end' }))

    await flushPromises()

    const context = FakeAudioContext.instances[0]
    expect(context.createBufferSource).toHaveBeenCalledTimes(1)
    expect(context.close).not.toHaveBeenCalled()

    await expect(session?.done).resolves.toBe('done')
    expect(context.close).toHaveBeenCalledTimes(1)
  })
})
