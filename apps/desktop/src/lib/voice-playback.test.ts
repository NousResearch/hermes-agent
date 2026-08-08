import { afterEach, describe, expect, it, vi } from 'vitest'

import { startSpeechStream } from './voice-playback'

class FakeWebSocket {
  static CONNECTING = 0
  static OPEN = 1
  static CLOSED = 3
  static instances: FakeWebSocket[] = []

  binaryType: BinaryType = 'blob'
  close = vi.fn(() => {
    this.readyState = FakeWebSocket.CLOSED
  })
  onclose: ((event: CloseEvent) => void) | null = null
  onerror: ((event: Event) => void) | null = null
  onmessage: ((event: MessageEvent) => void) | null = null
  onopen: ((event: Event) => void) | null = null
  readyState = FakeWebSocket.CONNECTING
  send = vi.fn()

  constructor(readonly url: string) {
    FakeWebSocket.instances.push(this)
  }
}

describe('voice playback streaming', () => {
  afterEach(() => {
    FakeWebSocket.instances = []
    vi.unstubAllGlobals()
    delete (window as { hermesDesktop?: unknown }).hermesDesktop
  })

  it('falls back to plain audio playback when WebAudio cannot initialize', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => undefined)
    vi.stubGlobal('WebSocket', FakeWebSocket)
    vi.stubGlobal(
      'AudioContext',
      class {
        constructor() {
          throw new Error('audio device unavailable')
        }
      }
    )
    ;(window as { hermesDesktop?: unknown }).hermesDesktop = {
      getConnection: vi.fn().mockResolvedValue({ authMode: 'token', wsUrl: 'ws://local/api/ws?token=abc' })
    }

    const session = await startSpeechStream({ source: 'read-aloud' })

    expect(session).not.toBeNull()
    const socket = FakeWebSocket.instances[0]

    expect(() => {
      socket.onmessage?.(
        new MessageEvent('message', {
          data: JSON.stringify({ channels: 1, sample_rate: 24_000, type: 'start' })
        })
      )
    }).not.toThrow()
    await expect(session?.done).resolves.toBe('fallback')
    expect(warn).toHaveBeenCalledWith(
      'Voice playback streaming disabled: AudioContext unavailable',
      expect.any(Error)
    )
  })
})
