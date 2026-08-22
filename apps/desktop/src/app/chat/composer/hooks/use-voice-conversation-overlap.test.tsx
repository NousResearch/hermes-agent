// Integration coverage for #91991: with the REAL voice-playback module, a
// barge-in followed by an out-of-order stream discovery (turn 1's stale start
// resolving after turn 2's session went live) must not kill the current reply,
// must not resurrect the interrupted one, and must leave every socket/context
// closed with no stale text feeding later turns.

import { act, cleanup, renderHook, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { BargeMonitorCallbacks } from '@/lib/voice-barge-in'
import { $voicePlayback } from '@/store/voice-playback'

import { useVoiceConversation } from './use-voice-conversation'

const control = vi.hoisted(() => {
  const pending: Array<() => void> = []

  return {
    getConnection: () =>
      new Promise<object>(resolve => {
        pending.push(() => resolve({ profile: 'test' }))
      }),
    pendingCount: () => pending.length,
    releaseLatest() {
      pending.pop()?.()
    }
  }
})

const bargeCalls: BargeMonitorCallbacks[] = []

vi.mock('@/lib/voice-barge-in', () => ({
  monitorSpeechDuringPlayback: (callbacks: BargeMonitorCallbacks) => {
    bargeCalls.push(callbacks)

    return vi.fn()
  }
}))

const micHandle = vi.hoisted(() => {
  let onSilence: null | (() => void) = null

  return {
    cancel: vi.fn(),
    handle: {
      cancel: vi.fn(),
      start: vi.fn(async (options: { onSilence: () => void }) => {
        onSilence = options.onSilence
      }),
      stop: vi.fn(async () => ({
        audio: new Blob(['voice'], { type: 'audio/webm' }),
        heardSpeech: true
      }))
    },
    triggerSilence() {
      onSilence?.()
    }
  }
})

vi.mock('./use-mic-recorder', () => ({
  useMicRecorder: () => ({ handle: micHandle.handle, level: 0 })
}))

vi.mock('@/lib/thinking-sound', () => ({
  startThinkingSound: vi.fn(),
  stopThinkingSound: vi.fn()
}))

vi.mock('@/store/notifications', () => ({
  notify: vi.fn(),
  notifyError: vi.fn()
}))

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      notifications: {
        voice: {
          configureSpeechToText: '',
          couldNotStartSession: '',
          microphoneFailed: '',
          playbackFailed: '',
          transcriptionFailed: '',
          unavailable: ''
        }
      }
    }
  })
}))

vi.mock('@hermes/shared', () => ({
  resolveGatewayWsUrl: async () => 'ws://gateway.test/api/ws'
}))

vi.mock('@/hermes', () => ({
  getApiRequestProfile: () => null,
  speakText: vi.fn(async () => ({
    data_url: 'data:audio/mpeg;base64,AAAA',
    mime_type: 'audio/mpeg',
    ok: true
  }))
}))

class FakeSocket {
  static readonly OPEN = 1
  static readonly CONNECTING = 0
  static instances: FakeSocket[] = []

  binaryType = 'blob'
  readyState = FakeSocket.CONNECTING
  sent: string[] = []
  url: string

  close = vi.fn(() => {
    this.readyState = 3
  })
  send = vi.fn((data: string) => {
    this.sent.push(data)
  })

  onopen: (() => void) | null = null
  onmessage: ((event: { data: string | ArrayBuffer }) => void) | null = null
  onclose: (() => void) | null = null
  onerror: (() => void) | null = null

  constructor(url: string) {
    this.url = url
    FakeSocket.instances.push(this)
  }

  serverOpen() {
    this.readyState = FakeSocket.OPEN
    this.onopen?.()
  }

  serverFrame(frame: object) {
    this.onmessage?.({ data: JSON.stringify(frame) })
  }

  serverPcm(bytes: number) {
    this.onmessage?.({ data: new ArrayBuffer(bytes) })
  }

  serverClose() {
    this.onclose?.()
  }
}

class FakeAudioContext {
  static instances: FakeAudioContext[] = []

  closed = false
  currentTime = 0
  destination = {}
  state = 'running'

  close = vi.fn(() => {
    this.closed = true

    return Promise.resolve()
  })
  resume = vi.fn(() => Promise.resolve())
  createBuffer = vi.fn((_channels: number, length: number, rate: number) => ({
    duration: length / rate,
    getChannelData: () => new Float32Array(length)
  }))
  createBufferSource = vi.fn(() => ({
    buffer: null as unknown,
    connect: vi.fn(),
    start: vi.fn()
  }))

  constructor() {
    FakeAudioContext.instances.push(this)
  }
}

interface PendingReply {
  id: string
  pending: boolean
  text: string
}

describe('useVoiceConversation overlapping stream discovery (#91991)', () => {
  beforeEach(() => {
    FakeSocket.instances = []
    FakeAudioContext.instances = []
    bargeCalls.length = 0
    vi.stubGlobal('WebSocket', FakeSocket)
    vi.stubGlobal('AudioContext', FakeAudioContext)
    vi.stubGlobal('hermesDesktop', { getConnection: control.getConnection })
    $voicePlayback.set({ audioElement: null, messageId: null, sequence: 0, source: null, status: 'idle' })
  })

  afterEach(() => {
    cleanup()
    vi.clearAllMocks()
    vi.unstubAllGlobals()
  })

  it('does not let a stale stream start kill the current reply or feed later turns', async () => {
    let response: PendingReply | null = null
    let transcriptions = 0

    const hook = renderHook(
      ({ enabled }) =>
        useVoiceConversation({
          busy: false,
          // Mirrors the real consumer: consuming removes the pending reply so
          // a later turn can never re-speak an already-consumed one.
          consumePendingResponse: () => {
            response = null
          },
          enabled,
          onSubmit: async (text: string) => {
            // The agent's reply only exists once the turn is in flight — never
            // during transcription. Mirrors the real submit/generate round trip.
            await new Promise(resolve => window.setTimeout(resolve, 0))

            response =
              text === 'start conversation'
                ? { id: 'reply-1', pending: false, text: 'First reply' }
                : { id: 'reply-2', pending: false, text: 'Second reply' }
          },
          onTranscribeAudio: async () =>
            transcriptions++ === 0 ? 'start conversation' : 'interrupting with a question',
          pendingResponse: () => response
        }),
      { initialProps: { enabled: false } }
    )

    hook.rerender({ enabled: true })

    await waitFor(() => expect(micHandle.handle.start).toHaveBeenCalled())

    await act(async () => {
      micHandle.triggerSilence()
    })

    // Turn 1's stream discovery is in flight (getConnection pending) when the
    // user barges in — the captured utterance submits turn 2.
    await waitFor(() => expect(control.pendingCount()).toBe(1))

    act(() => {
      bargeCalls.at(-1)?.onSpeech()
    })

    await act(async () => {
      bargeCalls.at(-1)?.onUtterance?.(new Blob(['interruption'], { type: 'audio/webm' }))
    })

    await waitFor(() => expect(control.pendingCount()).toBe(2))

    // Turn 2's discovery resolves first — its session goes live and speaks.
    act(() => {
      control.releaseLatest()
    })

    await waitFor(() => expect(FakeSocket.instances).toHaveLength(1))

    const currentSocket = FakeSocket.instances[0]

    currentSocket.serverOpen()
    currentSocket.serverFrame({ type: 'start', sample_rate: 24_000 })
    currentSocket.serverPcm(64)

    await waitFor(() => expect($voicePlayback.get().status).toBe('speaking'))

    // Turn 1's stale discovery resolves — it must not settle the current reply
    // and must not open a socket of its own.
    act(() => {
      control.releaseLatest()
    })

    await act(async () => {
      await new Promise(resolve => window.setTimeout(resolve, 0))
    })

    expect($voicePlayback.get().status).toBe('speaking')
    expect(currentSocket.close).not.toHaveBeenCalled()
    expect(FakeSocket.instances).toHaveLength(1)
    expect(hook.result.current.status).toBe('speaking')

    // Let the current reply finish naturally — the server streams the tail,
    // the session settles, and the conversation re-arms the microphone.
    currentSocket.serverFrame({ type: 'end' })

    await waitFor(() => expect(hook.result.current.status).toBe('listening'))

    // The turn's socket and audio context are closed and no stale text was
    // fed after the dust settled.
    expect(currentSocket.close).toHaveBeenCalled()
    expect(FakeAudioContext.instances.every(ctx => ctx.closed)).toBe(true)

    const sentAfterSettle = [...currentSocket.sent]

    await act(async () => {
      await new Promise(resolve => window.setTimeout(resolve, 400))
    })

    expect(currentSocket.sent).toEqual(sentAfterSettle)
  })
})
