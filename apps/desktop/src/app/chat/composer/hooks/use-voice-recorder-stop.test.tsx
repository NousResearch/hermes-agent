import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import type { MicRecording } from './use-mic-recorder'
import { useVoiceRecorder } from './use-voice-recorder'

const mic = vi.hoisted(() => ({ recording: false, start: vi.fn(), stop: vi.fn(), cancel: vi.fn() }))
vi.mock('./use-mic-recorder', () => ({ useMicRecorder: () => ({ handle: mic, recording: mic.recording, level: 0 }) }))
vi.mock('@/lib/stt-lease', () => ({ syncSttLease: async () => {}, VOICE_INPUT_LEASE: 'test' }))
vi.mock('@/lib/voice-client-direct', () => ({ fetchVoiceClientConfigFor: async () => null }))
vi.mock('@/store/notifications', () => ({ notify: vi.fn(), notifyError: vi.fn() }))
vi.mock('@/store/desktop-metrics', () => ({ recordFeatureUse: vi.fn() }))
vi.mock('@/hermes', async original => ({
  ...(await original<object>()),
  resolveOwnerNow: () => ({ connectionId: 'test', profile: 'default' })
}))
const audio = { audio: new Blob(), durationMs: 500, heardSpeech: true } as MicRecording
beforeEach(() => {
  vi.clearAllMocks()
  mic.stop.mockReset()
  mic.recording = false
  mic.start.mockImplementation(async () => {
    mic.recording = true
  })
  mic.stop.mockResolvedValue(audio)
})
afterEach(cleanup)

function probe() {
  const transcript = vi.fn(),
    transcribe = vi.fn(async () => 'Finished dictation')

  const hook = renderHook(() =>
    useVoiceRecorder({
      maxRecordingSeconds: 60,
      onTranscribeAudio: transcribe,
      onTranscript: transcript,
      focusInput: vi.fn()
    })
  )

  return { hook, transcript, transcribe }
}

it('delivers the final transcript through onTranscript', async () => {
  const p = probe()
  await act(async () => p.hook.result.current.dictate())
  await act(async () => p.hook.result.current.dictate())
  expect(p.transcript).toHaveBeenCalledWith('Finished dictation')
})
it('a failed transcription produces no completion text', async () => {
  const p = probe()
  p.transcribe.mockRejectedValue(new Error('STT failed'))
  await act(async () => p.hook.result.current.dictate())
  await act(async () => p.hook.result.current.dictate())
  expect(p.transcript).not.toHaveBeenCalled()
})
it('completes one transcript when stop is requested twice before the recorder settles', async () => {
  const p = probe()
  await act(async () => p.hook.result.current.dictate())
  let finish!: (value: MicRecording) => void
  // useMicRecorder's stop() returns null once MediaRecorder is inactive, even
  // while its first stop result is pending. No real device is opened here.
  mic.stop
    .mockReturnValueOnce(
      new Promise(resolve => {
        finish = resolve
      })
    )
    .mockResolvedValueOnce(null)
  const dictate = p.hook.result.current.dictate
  await act(async () => {
    dictate()
    dictate()
  })
  await act(async () => finish(audio))
  expect(mic.stop).toHaveBeenCalledOnce()
  expect(p.transcript).toHaveBeenCalledExactlyOnceWith('Finished dictation')
})

it('the time limit and a simultaneous manual stop complete only once', async () => {
  vi.useFakeTimers()

  try {
    const p = probe()
    await act(async () => p.hook.result.current.dictate())
    let finish!: (value: MicRecording) => void
    mic.stop.mockReturnValueOnce(new Promise(resolve => { finish = resolve }))
    await act(async () => vi.advanceTimersByTimeAsync(60_000))
    await act(async () => p.hook.result.current.dictate())
    await act(async () => finish(audio))
    expect(mic.stop).toHaveBeenCalledOnce()
    expect(p.transcript).toHaveBeenCalledExactlyOnceWith('Finished dictation')
  } finally { vi.useRealTimers() }
})

it('a later recording has its own stop guard and completes normally', async () => {
  const p = probe()
  await act(async () => p.hook.result.current.dictate())
  await act(async () => p.hook.result.current.dictate())
  mic.recording = false
  p.hook.rerender()
  await act(async () => p.hook.result.current.dictate())
  await act(async () => p.hook.result.current.dictate())
  expect(p.transcript).toHaveBeenCalledTimes(2)
})
