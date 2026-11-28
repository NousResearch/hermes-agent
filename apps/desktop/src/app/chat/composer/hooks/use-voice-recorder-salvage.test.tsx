import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { MicRecording } from './use-mic-recorder'
import { useVoiceRecorder } from './use-voice-recorder'

// #149000: switching chat or bot while a dictation is still transcribing used to
// drop the transcript silently — the composer unmounts, `dictationRef` is
// cleared, and every `live()` guard returned early. The text now goes to
// `onSalvaged` (the session's persisted draft) instead.

let recording = false

const micHandle = {
  cancel: vi.fn(),
  start: vi.fn(async () => {
    recording = true
  }),
  stop: vi.fn<() => Promise<MicRecording | null>>(async () => {
    recording = false

    return { audio: new Blob(), durationMs: 500, heardSpeech: true }
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

vi.mock('@/store/notifications', () => ({
  notify: vi.fn(),
  notifyError: vi.fn()
}))

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

vi.mock('@/hermes', () => ({
  resolveOwnerNow: () => ({ connectionId: 'gateway-a', profile: 'worker_alpha' })
}))

describe('useVoiceRecorder transcript salvage', () => {
  beforeEach(() => {
    cleanup()
    recording = false
    micHandle.start.mockClear()
    micHandle.stop.mockReset()
    micHandle.stop.mockImplementation(async () => {
      recording = false

      return { audio: new Blob(), durationMs: 500, heardSpeech: true }
    })
  })

  afterEach(() => {
    cleanup()
  })

  function renderRecorder(
    onTranscribeAudio: (audio: Blob, owner?: unknown) => Promise<string>,
    handlers: { onSalvaged?: (text: string) => void; onTranscript?: (text: string) => void } = {}
  ) {
    return renderHook(() =>
      useVoiceRecorder({
        focusInput: () => undefined,
        maxRecordingSeconds: 30,
        onSalvaged: handlers.onSalvaged,
        onTranscript: handlers.onTranscript ?? (() => undefined),
        onTranscribeAudio
      })
    )
  }

  it('parks the transcript when the composer unmounts mid-transcription', async () => {
    const pending = Promise.withResolvers<string>()
    const onSalvaged = vi.fn()
    const onTranscript = vi.fn()
    const hook = renderRecorder(() => pending.promise, { onSalvaged, onTranscript })

    // Record, then stop: transcription starts and stays on the wire.
    await act(async () => {
      hook.result.current.dictate()
    })
    await act(async () => {
      hook.result.current.dictate()
    })

    // The user switches chat/bot: the composer unmounts before the text lands.
    hook.unmount()

    await act(async () => {
      pending.resolve('转写好的话')
      await Promise.resolve()
    })

    expect(onTranscript).not.toHaveBeenCalled()
    expect(onSalvaged).toHaveBeenCalledExactlyOnceWith('转写好的话')
  })

  it('still inserts the transcript when the composer is alive', async () => {
    const onSalvaged = vi.fn()
    const onTranscript = vi.fn()
    const hook = renderRecorder(async () => '   还活着   ', { onSalvaged, onTranscript })

    await act(async () => {
      hook.result.current.dictate()
    })
    await act(async () => {
      hook.result.current.dictate()
    })

    expect(onSalvaged).not.toHaveBeenCalled()
    expect(onTranscript).toHaveBeenCalledExactlyOnceWith('还活着')
  })

  it('does not salvage an empty transcript', async () => {
    const onSalvaged = vi.fn()
    const pending = Promise.withResolvers<string>()
    const hook = renderRecorder(() => pending.promise, { onSalvaged })

    await act(async () => {
      hook.result.current.dictate()
    })
    await act(async () => {
      hook.result.current.dictate()
    })
    hook.unmount()

    await act(async () => {
      pending.resolve('   ')
      await Promise.resolve()
    })

    expect(onSalvaged).not.toHaveBeenCalled()
  })
})
