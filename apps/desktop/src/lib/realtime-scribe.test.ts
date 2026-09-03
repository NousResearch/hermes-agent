import { afterEach, describe, expect, it, vi } from 'vitest'

import { setApiRequestConnection, setApiRequestProfile } from '@/hermes'

const { connect } = vi.hoisted(() => ({ connect: vi.fn() }))

vi.mock('@elevenlabs/client', () => ({
  AudioFormat: { PCM_16000: 'pcm_16000' },
  CommitStrategy: { VAD: 'vad' },
  RealtimeEvents: {
    CLOSE: 'close',
    COMMITTED_TRANSCRIPT: 'committed_transcript',
    ERROR: 'error',
    OPEN: 'open',
    PARTIAL_TRANSCRIPT: 'partial_transcript'
  },
  Scribe: { connect }
}))

import { startRealtimeScribe } from './realtime-scribe'

function mockDesktopApi(response: unknown) {
  const api = vi.fn(async () => response)
  Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { api } })

  return api
}

describe('startRealtimeScribe', () => {
  afterEach(() => {
    connect.mockReset()
    setApiRequestConnection(null)
    setApiRequestProfile(null)
    Reflect.deleteProperty(window, 'hermesDesktop')
  })

  it('mints a scoped token and opens a VAD microphone connection', async () => {
    const listeners = new Map<string, (data?: unknown) => void>()
    let muted = false

    const connection = {
      close: vi.fn(),
      commit: vi.fn(),
      get isMuted() {
        return muted
      },
      mute: vi.fn(() => {
        muted = true
      }),
      on: vi.fn((event: string, listener: (data?: unknown) => void) => listeners.set(event, listener)),
      unmute: vi.fn(() => {
        muted = false
      })
    }

    connect.mockReturnValue(connection)

    const api = mockDesktopApi({
      ok: true,
      token: 'sutkn_one_use',
      websocket_url: 'wss://api.elevenlabs.io',
      model: 'scribe_v2_realtime',
      language: 'en'
    })

    setApiRequestConnection('remote')
    setApiRequestProfile('default')
    const committed = vi.fn()

    const pending = startRealtimeScribe({ onCommitted: committed })
    await vi.waitFor(() => expect(listeners.has('open')).toBe(true))
    listeners.get('open')?.()
    const session = await pending

    expect(api).toHaveBeenCalledWith(
      expect.objectContaining({ connectionId: 'remote', path: '/api/audio/scribe-token', profile: 'default' })
    )
    expect(connect).toHaveBeenCalledWith(
      expect.objectContaining({
        token: 'sutkn_one_use',
        modelId: 'scribe_v2_realtime',
        baseUri: 'wss://api.elevenlabs.io',
        commitStrategy: 'vad',
        vadSilenceThresholdSecs: 1,
        microphone: expect.objectContaining({ echoCancellation: true, noiseSuppression: true })
      })
    )

    listeners.get('committed_transcript')?.({ text: '  hello there  ' })
    expect(committed).toHaveBeenCalledWith('hello there')
    session?.mute()
    session?.unmute()
    session?.commit()
    session?.close()
    expect(connection.mute).toHaveBeenCalled()
    expect(connection.unmute).toHaveBeenCalled()
    expect(connection.commit).toHaveBeenCalled()
    expect(connection.close).toHaveBeenCalled()
  })

  it('returns null when the backend cannot mint a token', async () => {
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { api: vi.fn(async () => Promise.reject(new Error('409'))) }
    })

    await expect(startRealtimeScribe({ onCommitted: vi.fn() })).resolves.toBeNull()
    expect(connect).not.toHaveBeenCalled()
  })
})
