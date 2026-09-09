import { afterEach, expect, it, vi } from 'vitest'

const { notifyError } = vi.hoisted(() => ({ notifyError: vi.fn() }))
vi.mock('@/store/notifications', () => ({ notifyError }))
vi.mock('@/lib/voice-client-direct', () => ({ directTtsConfig: async () => null }))
vi.mock('@/hermes', () => ({
  getApiRequestConnection: () => null,
  getApiRequestProfile: () => 'voice',
  speakText: vi.fn()
}))

import { speakText } from '@/hermes'

import { startSpeechStream, stopVoicePlayback } from './voice-playback'

class Socket {
  static OPEN = 1
  static CONNECTING = 0
  static latest: Socket
  readyState = 1
  binaryType = ''
  onmessage: ((event: { data: string | ArrayBuffer }) => void) | null = null
  onopen: (() => void) | null = null
  onclose: (() => void) | null = null
  send = vi.fn()
  close = vi.fn()
  constructor() {
    Socket.latest = this
  }
}

afterEach(() => {
  stopVoicePlayback()
  vi.clearAllMocks()
  vi.unstubAllGlobals()
  Reflect.deleteProperty(window, 'hermesDesktop')
})

it.each([false, true])('surfaces provider failure without replay (audio started=%s)', async partial => {
  const close = vi.fn(async () => undefined)
  vi.stubGlobal('WebSocket', Socket)
  vi.stubGlobal(
    'AudioContext',
    class {
      state = 'running'
      currentTime = 0
      destination = {}
      close = close
      createBuffer(_channels: number, length: number, rate: number) {
        return { duration: length / rate, getChannelData: () => new Float32Array(length) }
      }
      createBufferSource() {
        return { buffer: null, connect: vi.fn(), start: vi.fn() }
      }
    }
  )
  Object.defineProperty(window, 'hermesDesktop', {
    configurable: true,
    value: { getConnection: async () => ({ authMode: 'token', wsUrl: 'ws://localhost:1234/api/ws?token=test' }) }
  })

  const session = await startSpeechStream({ source: 'voice-conversation' })
  expect(session).not.toBeNull()
  Socket.latest.onmessage?.({ data: JSON.stringify({ type: 'start', sample_rate: 24000, channels: 1 }) })

  if (partial) {
    Socket.latest.onmessage?.({ data: new Int16Array([10, 20]).buffer })
  }
  Socket.latest.onmessage?.({ data: JSON.stringify({ type: 'error', message: 'Speech synthesis failed.' }) })

  await expect(session!.done).resolves.toBe('done')
  expect(notifyError).toHaveBeenCalledOnce()
  expect(Socket.latest.close).toHaveBeenCalledOnce()
  expect(close).toHaveBeenCalledOnce()
  expect(speakText).not.toHaveBeenCalled()
})
