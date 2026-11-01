import { act, cleanup, renderHook, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { NotificationAction } from '@/store/notifications'

import type { MicRecording } from './use-mic-recorder'
import { useVoiceRecorder } from './use-voice-recorder'

// A dictation take is the user's words in the only form they exist. When its
// transcription fails (provider busy, timeout, dropped connection), the failure
// must end in a way back to the recording, never in a dead toast: a five-minute
// take cannot be "tried again" by speaking it again.

let recording = false

const take: MicRecording = {
  audio: new Blob(['five minutes of speech'], { type: 'audio/webm' }),
  durationMs: 300_000,
  heardSpeech: true
}

const micHandle = {
  cancel: vi.fn(),
  start: vi.fn(async () => {
    recording = true
  }),
  stop: vi.fn<() => Promise<MicRecording | null>>(async () => {
    recording = false

    return take
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

vi.mock('@/lib/stt-lease', () => ({
  syncSttLease: async () => undefined,
  VOICE_INPUT_LEASE: 'desktop:voice-input:test'
}))

interface ErrorToastOptions {
  action?: NotificationAction
  secondaryAction?: NotificationAction
}

const toasts = vi.hoisted(() => ({
  notify: vi.fn((_input: { action?: NotificationAction }) => 'toast'),
  notifyError: vi.fn((_error: unknown, _fallback: string, _options?: ErrorToastOptions) => 'toast')
}))

vi.mock('@/store/notifications', () => ({
  notify: toasts.notify,
  notifyError: toasts.notifyError
}))

const downloads = vi.hoisted(() => ({ downloadBlob: vi.fn((_blob: Blob, _name: string) => undefined) }))

vi.mock('@/lib/download', () => ({ downloadBlob: downloads.downloadBlob }))

vi.mock('@/store/desktop-metrics', () => ({
  recordFeatureUse: vi.fn()
}))

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      notifications: {
        voice: {
          noSpeechDetected: 'no speech',
          recordingFailed: 'recording failed',
          retryTranscription: 'retry',
          saveRecording: 'save recording',
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
  useComposerScope: () => ({ connectionId: undefined, profile: undefined, target: 'main' })
}))

vi.mock('@/hermes', () => ({
  resolveOwnerNow: () => ({ connectionId: 'gateway-a', profile: 'default' })
}))

// The latest way back the failure offered: the error toast's two buttons.
function lastOffer(): { retry?: NotificationAction; save?: NotificationAction } {
  const options = toasts.notifyError.mock.calls.at(-1)?.[2]

  return { retry: options?.action, save: options?.secondaryAction }
}

function renderRecorder(transcribe: (audio: Blob, owner?: unknown) => Promise<string>) {
  const onTranscript = vi.fn()

  const hook = renderHook(() =>
    useVoiceRecorder({
      focusInput: () => undefined,
      maxRecordingSeconds: 600,
      onTranscript,
      onTranscribeAudio: transcribe
    })
  )

  return { hook, onTranscript }
}

async function recordOneTake(hook: ReturnType<typeof renderRecorder>['hook']) {
  await act(async () => {
    hook.result.current.dictate() // mic open
  })
  await act(async () => {
    await Promise.resolve()
  })
  await act(async () => {
    hook.result.current.dictate() // stop → transcription
  })
}

const providerBusy = () => new Error('503 Service Unavailable: the service is experiencing high traffic')

describe('useVoiceRecorder keeps a take whose transcription failed', () => {
  beforeEach(() => {
    cleanup()
    recording = false
    toasts.notify.mockClear()
    toasts.notifyError.mockClear()
    downloads.downloadBlob.mockClear()
  })

  afterEach(() => {
    cleanup()
  })

  it('offers a retry on the same recording that delivers its words, and the file itself', async () => {
    const transcribe = vi
      .fn<(audio: Blob, owner?: unknown) => Promise<string>>()
      .mockRejectedValueOnce(providerBusy())
      .mockResolvedValueOnce('everything I said for five minutes')

    const { hook, onTranscript } = renderRecorder(transcribe)

    await recordOneTake(hook)

    await waitFor(() => expect(transcribe).toHaveBeenCalledTimes(1))
    await waitFor(() => expect(hook.result.current.voiceStatus).toBe('idle'))
    expect(onTranscript).not.toHaveBeenCalled()

    const { retry, save } = lastOffer()

    expect(retry, 'a failed transcription must leave a way back to the recording').toBeDefined()

    save!.onClick()
    expect(downloads.downloadBlob.mock.calls[0]?.[0]).toBe(take.audio)

    await act(async () => {
      retry!.onClick()
    })

    await waitFor(() => expect(transcribe).toHaveBeenCalledTimes(2))
    expect(transcribe.mock.calls[1][0]).toBe(take.audio)
    await waitFor(() => expect(onTranscript).toHaveBeenCalledWith('everything I said for five minutes'))
  })

  it('keeps the take across a composer remount and delivers it to the composer mounted now', async () => {
    const transcribe = vi
      .fn<(audio: Blob, owner?: unknown) => Promise<string>>()
      .mockRejectedValueOnce(providerBusy())
      .mockResolvedValueOnce('words from before Settings opened')

    const first = renderRecorder(transcribe)

    await recordOneTake(first.hook)
    await waitFor(() => expect(lastOffer().retry).toBeDefined())

    // Settings opens: the chat bar unmounts, the notice stays.
    first.hook.unmount()

    await act(async () => {
      lastOffer().retry!.onClick()
    })

    // Nothing can receive the words yet: the take is offered again, untouched.
    expect(transcribe).toHaveBeenCalledTimes(1)
    expect(toasts.notifyError).toHaveBeenCalledTimes(2)

    // Back to the chat: a new composer instance mounts.
    const second = renderRecorder(transcribe)

    await act(async () => {
      lastOffer().retry!.onClick()
    })

    await waitFor(() => expect(transcribe).toHaveBeenCalledTimes(2))
    expect(transcribe.mock.calls[1][0]).toBe(take.audio)
    await waitFor(() => expect(second.onTranscript).toHaveBeenCalledWith('words from before Settings opened'))
    expect(first.onTranscript).not.toHaveBeenCalled()
  })
})
