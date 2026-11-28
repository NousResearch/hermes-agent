import { act, cleanup, renderHook, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { registry } from '@/contrib/registry'

import { COMPOSER_AREAS, type ComposerDictationTake, type ComposerDictationTranscriber } from '../contrib'

import type { MicRecorderOptions, MicRecording } from './use-mic-recorder'
import { useVoiceRecorder } from './use-voice-recorder'

// A `composer.dictationTranscriber` provider decides a take when its mic opens. A take it
// claims opens no live provider stream (no PCM tap), takes no backend warm-up
// lease and never reaches the app's own transcription path unless the plugin
// relays it; a take nobody claims goes down the app's own path exactly as before.

let recording = false
let micOptions: MicRecorderOptions | null = null

const micHandle = {
  cancel: vi.fn(),
  start: vi.fn(async (options: MicRecorderOptions) => {
    micOptions = options
    recording = true
  }),
  stop: vi.fn<() => Promise<MicRecording | null>>(async () => {
    recording = false

    return { audio: new Blob(['clip'], { type: 'audio/webm' }), durationMs: 900, heardSpeech: true }
  })
}

vi.mock('./use-mic-recorder', () => ({
  useMicRecorder: () => ({
    handle: micHandle,
    level: 0,
    get recording() {
      return recording
    }
  })
}))

const backend = vi.hoisted(() => ({ leases: [] as unknown[][], requests: [] as string[] }))

vi.mock('@/lib/stt-lease', () => ({
  syncSttLease: vi.fn(async (...args: unknown[]) => {
    backend.leases.push(args)
  }),
  VOICE_INPUT_LEASE: 'desktop:voice-input:test'
}))

vi.mock('@/lib/voice-stream', () => ({
  openDictationStream: vi.fn(async () => {
    backend.requests.push('live-stream')

    return null
  })
}))

vi.mock('@/lib/voice-client-direct', () => ({
  fetchVoiceClientConfigFor: vi.fn(async () => {
    backend.requests.push('voice-config')

    return { stt: { streaming: true } }
  })
}))

const notifications = vi.hoisted(() => ({ notify: vi.fn(), notifyError: vi.fn() }))

vi.mock('@/store/notifications', () => notifications)

vi.mock('@/store/desktop-metrics', () => ({ recordFeatureUse: vi.fn() }))

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      notifications: {
        voice: {
          noSpeechDetected: 'no speech',
          recordingFailed: 'recording failed',
          transcriptionFailed: 'transcription failed',
          transcriptionUnavailable: 'unavailable',
          tryRecordingAgain: 'try again',
          unavailable: 'unavailable'
        }
      }
    }
  })
}))

vi.mock('../scope', () => ({
  useComposerScope: () => ({ connectionId: undefined, profile: undefined })
}))

const OWNER = { connectionId: 'gateway-a', profile: 'worker_alpha' }

vi.mock('@/hermes', () => ({
  resolveOwnerNow: () => OWNER
}))

let disposers: Array<() => void> = []

function provide(provider: Partial<ComposerDictationTranscriber>, id = 'on-device', order?: number) {
  disposers.push(
    registry.register({ area: COMPOSER_AREAS.dictationTranscriber, data: provider, id, order, source: 'plugin:test' })
  )
}

function renderRecorder(onTranscribeAudio: (audio: Blob, owner?: unknown) => Promise<string>) {
  const heard: string[] = []

  const hook = renderHook(() =>
    useVoiceRecorder({
      focusInput: () => undefined,
      maxRecordingSeconds: 30,
      onTranscript: text => heard.push(text),
      onTranscribeAudio
    })
  )

  return { heard, hook }
}

async function openMic(hook: ReturnType<typeof renderRecorder>['hook']) {
  await act(async () => hook.result.current.dictate())
  await waitFor(() => expect(hook.result.current.voiceStatus).toBe('recording'))
}

async function closeMic(hook: ReturnType<typeof renderRecorder>['hook']) {
  await act(async () => hook.result.current.dictate())
  await waitFor(() => expect(hook.result.current.voiceStatus).toBe('idle'))
}

describe('useVoiceRecorder with a composer.dictationTranscriber provider', () => {
  beforeEach(() => {
    cleanup()
    recording = false
    micOptions = null
    backend.leases.length = 0
    backend.requests.length = 0
    notifications.notify.mockClear()
    notifications.notifyError.mockClear()
  })

  afterEach(() => {
    cleanup()
    disposers.forEach(dispose => dispose())
    disposers = []
  })

  it('a claimed take is transcribed by the plugin alone: no PCM tap, no lease, nothing to the backend', async () => {
    const transcribe = vi.fn(async () => '  Ship it Friday.  ')
    provide({ claims: () => true, transcribe })
    const appPath = vi.fn(async () => 'from the backend')
    const { heard, hook } = renderRecorder(appPath)

    await openMic(hook)
    expect(micOptions?.onPcm).toBeUndefined()
    expect(micOptions?.onPcmRate).toBeUndefined()
    await closeMic(hook)

    expect(heard).toEqual(['Ship it Friday.'])
    expect(transcribe).toHaveBeenCalledTimes(1)
    expect(appPath).not.toHaveBeenCalled()
    expect(backend.requests).toEqual([])
    expect(backend.leases).toEqual([])
  })

  it('a take the plugin could not transcribe warns, inserts nothing and is not relayed', async () => {
    provide({ claims: () => true, transcribe: async () => null })
    const appPath = vi.fn(async () => 'from the backend')
    const { heard, hook } = renderRecorder(appPath)

    await openMic(hook)
    await closeMic(hook)

    expect(heard).toEqual([])
    expect(appPath).not.toHaveBeenCalled()
    expect(notifications.notify).toHaveBeenCalledWith(
      expect.objectContaining({ kind: 'warning', message: 'try again', title: 'transcription failed' })
    )
    expect(backend.leases).toEqual([])
  })

  it('a rejected transcription is reported like any transcription failure', async () => {
    provide({
      claims: () => true,
      transcribe: async () => {
        throw new Error('model unloadable')
      }
    })
    const { heard, hook } = renderRecorder(vi.fn(async () => 'from the backend'))

    await openMic(hook)
    await closeMic(hook)

    expect(heard).toEqual([])
    expect(notifications.notifyError).toHaveBeenCalledWith(expect.any(Error), 'transcription failed')
  })

  it("relay() sends the take's audio down the app's own path on the recording's owner", async () => {
    let take: ComposerDictationTake | null = null
    provide({
      claims: () => true,
      transcribe: async (_audio, given) => {
        take = given

        return given.relay()
      }
    })
    const appPath = vi.fn(async () => 'from the backend')
    const { heard, hook } = renderRecorder(appPath)

    await openMic(hook)
    await closeMic(hook)

    expect(take).not.toBeNull()
    expect(appPath).toHaveBeenCalledWith(expect.any(Blob), OWNER)
    expect(heard).toEqual(['from the backend'])
  })

  it("a declining, throwing or malformed provider leaves the take on the app's own path", async () => {
    provide({ claims: () => false, transcribe: vi.fn() }, 'declines', 1)
    provide(
      {
        claims: () => {
          throw new Error('broken plugin')
        },
        transcribe: vi.fn()
      },
      'throws',
      2
    )
    provide({ claims: () => true }, 'malformed', 3)
    const appPath = vi.fn(async () => '  from the backend ')
    const { heard, hook } = renderRecorder(appPath)

    await openMic(hook)
    micOptions?.onPcmRate?.(48_000)
    await closeMic(hook)

    expect(heard).toEqual(['from the backend'])
    expect(appPath).toHaveBeenCalledWith(expect.any(Blob), OWNER)
    expect(backend.requests).toEqual(expect.arrayContaining(['voice-config', 'live-stream']))
    expect(backend.leases.map(args => args[1])).toEqual([true, false])
  })

  it('the first claiming provider in registry order owns the take', async () => {
    const later = vi.fn(async () => 'later')
    const first = vi.fn(async () => 'first')
    provide({ claims: () => true, transcribe: later }, 'later', 2)
    provide({ claims: () => true, transcribe: first }, 'first', 1)
    const { heard, hook } = renderRecorder(vi.fn(async () => 'from the backend'))

    await openMic(hook)
    await closeMic(hook)

    expect(heard).toEqual(['first'])
    expect(later).not.toHaveBeenCalled()
  })

  it('the take is decided at mic-open: a provider claiming mid-recording does not take it', async () => {
    let ready = false
    const transcribe = vi.fn(async () => 'on device')
    provide({ claims: () => ready, transcribe })
    const appPath = vi.fn(async () => 'from the backend')
    const { heard, hook } = renderRecorder(appPath)

    await openMic(hook)
    ready = true
    await closeMic(hook)

    expect(heard).toEqual(['from the backend'])
    expect(transcribe).not.toHaveBeenCalled()
    expect(backend.leases.map(args => args[1])).toEqual([true, false])
  })
})
