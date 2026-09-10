import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const { directTtsConfig } = vi.hoisted(() => ({ directTtsConfig: vi.fn(async () => null) }))
vi.mock('@/lib/voice-client-direct', () => ({ directTtsConfig }))
vi.mock('@/hermes', () => ({
  getApiRequestConnection: () => null,
  getApiRequestProfile: () => 'voice',
  speakText: vi.fn()
}))

import { $voicePlayback } from '@/store/voice-playback'

import { startSpeechStream, stopVoicePlayback } from './voice-playback'

class Socket {
  static OPEN = 1
  static CONNECTING = 0
  static instances: Socket[] = []
  readyState = Socket.CONNECTING
  binaryType = ''
  onmessage: ((event: { data: string | ArrayBuffer }) => void) | null = null
  onopen: (() => void) | null = null
  onclose: (() => void) | null = null
  send = vi.fn()
  close = vi.fn()

  constructor() {
    Socket.instances.push(this)
  }

  open() {
    this.readyState = Socket.OPEN
    this.onopen?.()
  }

  frame(data: object | ArrayBuffer) {
    this.onmessage?.({ data: data instanceof ArrayBuffer ? data : JSON.stringify(data) })
  }
}

class PlaybackContext {
  static instances: PlaybackContext[] = []
  state = 'running'
  destination = {}
  close = vi.fn(async () => undefined)
  private openedAt = Date.now()

  constructor() {
    PlaybackContext.instances.push(this)
  }

  get currentTime() {
    return (Date.now() - this.openedAt) / 1_000
  }

  createBuffer(_channels: number, length: number, rate: number) {
    return { duration: length / rate, getChannelData: () => new Float32Array(length) }
  }

  createBufferSource() {
    return { buffer: null, connect: vi.fn(), start: vi.fn() }
  }
}

beforeEach(() => {
  vi.useFakeTimers()
  Socket.instances = []
  PlaybackContext.instances = []
  directTtsConfig.mockResolvedValue(null)
  vi.stubGlobal('WebSocket', Socket)
  vi.stubGlobal('AudioContext', PlaybackContext)
  Object.defineProperty(window, 'hermesDesktop', {
    configurable: true,
    value: { getConnection: async () => ({ authMode: 'token', wsUrl: 'ws://localhost:1234/api/ws?token=test' }) }
  })
})

afterEach(() => {
  stopVoicePlayback()
  vi.clearAllMocks()
  vi.useRealTimers()
  vi.unstubAllGlobals()
  Reflect.deleteProperty(window, 'hermesDesktop')
})

describe('speech playback ownership', () => {
  it('does not resurrect a stopped setup or let an older setup replace a newer reply', async () => {
    let complete!: (value: null) => void
    directTtsConfig.mockImplementationOnce(() => new Promise(resolve => (complete = resolve)))
    const stopped = startSpeechStream({ source: 'voice-conversation' })
    stopVoicePlayback()
    complete(null)
    await expect(stopped).resolves.toBeNull()
    expect(Socket.instances).toHaveLength(0)

    directTtsConfig.mockImplementationOnce(() => new Promise(resolve => (complete = resolve)))
    const older = startSpeechStream({ source: 'voice-conversation', messageId: 'older' })
    const newer = await startSpeechStream({ source: 'voice-conversation', messageId: 'newer' })
    complete(null)
    await expect(older).resolves.toBeNull()
    expect(newer).not.toBeNull()
    expect(Socket.instances).toHaveLength(1)
    expect($voicePlayback.get()).toMatchObject({ messageId: 'newer', status: 'preparing' })
  })

  it('ignores stopped socket events and old completion while a replacement is speaking', async () => {
    const older = await startSpeechStream({ source: 'voice-conversation', messageId: 'older' })
    const oldSocket = Socket.instances[0]
    oldSocket.open()
    stopVoicePlayback()
    oldSocket.frame({ type: 'start', sample_rate: 24_000 })
    oldSocket.frame(new Int16Array([10, 20]).buffer)
    expect(PlaybackContext.instances).toHaveLength(0)

    const newer = await startSpeechStream({ source: 'voice-conversation', messageId: 'newer' })
    const socket = Socket.instances[1]
    socket.open()
    socket.frame({ type: 'start', sample_rate: 24_000 })
    socket.frame(new Int16Array([10, 20]).buffer)
    oldSocket.onclose?.()
    await older!.done
    await vi.advanceTimersByTimeAsync(200)
    expect($voicePlayback.get()).toMatchObject({ messageId: 'newer', status: 'speaking' })
    expect(newer).not.toBeNull()
  })
})

describe('speech stream recovery', () => {
  it('falls back when the socket never connects or a finished reply never receives audio', async () => {
    const unopened = await startSpeechStream({ source: 'voice-conversation' })
    const unopenedOutcome = vi.fn()
    void unopened!.done.then(unopenedOutcome)
    await vi.advanceTimersByTimeAsync(240_000)
    expect(unopenedOutcome).toHaveBeenCalledWith('fallback')
    expect(Socket.instances[0].close).toHaveBeenCalledOnce()

    const silent = await startSpeechStream({ source: 'voice-conversation' })
    const socket = Socket.instances[1]
    socket.open()
    socket.frame({ type: 'start', sample_rate: 24_000 })
    silent!.append('This completed reply must not leave the microphone stuck forever.')
    silent!.finish()
    const silentOutcome = vi.fn()
    void silent!.done.then(silentOutcome)
    await vi.advanceTimersByTimeAsync(120_000)
    expect(silentOutcome).not.toHaveBeenCalled()
    await vi.advanceTimersByTimeAsync(120_000)
    expect(silentOutcome).toHaveBeenCalledWith('fallback')
    expect(socket.close).toHaveBeenCalledOnce()
  })

  it('allows buffered audio to drain before recovering a missing terminal frame', async () => {
    const session = await startSpeechStream({ source: 'voice-conversation' })
    const socket = Socket.instances[0]
    socket.open()
    socket.frame({ type: 'start', sample_rate: 24_000 })
    session!.append('A long response is still valid while its queued audio is playing.')
    session!.finish()
    socket.frame(new Int16Array(24_000 * 90).buffer)
    const outcome = vi.fn()
    void session!.done.then(outcome)
    await vi.advanceTimersByTimeAsync(80_000)
    expect(outcome).not.toHaveBeenCalled()
    await vi.advanceTimersByTimeAsync(240_000)
    expect(outcome).toHaveBeenCalledWith('done')
  })
})
