import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { useHermesConfig } from '@/app/session/hooks/use-hermes-config'
import { getHermesConfig } from '@/hermes'

import type { MicRecording } from './use-mic-recorder'
import { useVoiceRecorder } from './use-voice-recorder'

// #154537: `voice.max_recording_seconds` reached only the main chat's composer
// (threaded as a prop), so a session tile's dictation stopped at the built-in
// 120 s whatever the config said. The cap is a config value like the other
// voice settings: every composer's recorder must honour it once config loads.

let recording = false

const micHandle = {
  cancel: vi.fn(),
  start: vi.fn(async () => {
    recording = true
  }),
  stop: vi.fn<() => Promise<MicRecording | null>>(async () => {
    recording = false

    return null
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
  syncSttLease: vi.fn(async () => undefined),
  VOICE_INPUT_LEASE: 'desktop:voice-input:test'
}))

vi.mock('@/store/notifications', () => ({ notify: vi.fn(), notifyError: vi.fn() }))
vi.mock('@/store/desktop-metrics', () => ({ recordFeatureUse: vi.fn() }))

vi.mock('@/i18n', () => ({
  useI18n: () => ({ t: { notifications: { voice: {} } } })
}))

vi.mock('../scope', () => ({
  useComposerScope: () => ({ connectionId: undefined, profile: undefined })
}))

vi.mock('@/hermes', () => ({
  getHermesConfig: vi.fn(),
  getHermesConfigDefaults: vi.fn().mockResolvedValue({}),
  resolveOwnerNow: () => ({ connectionId: 'gateway-a', profile: 'default' }),
  setApiRequestProfile: vi.fn()
}))

describe('dictation cap from voice.max_recording_seconds', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    recording = false
    micHandle.start.mockClear()
    micHandle.stop.mockClear()
  })

  afterEach(() => {
    cleanup()
    vi.useRealTimers()
  })

  it('a composer that is not the main chat still records up to the configured cap', async () => {
    vi.mocked(getHermesConfig).mockResolvedValue({ voice: { max_recording_seconds: 600 } } as never)
    const config = renderHook(() => useHermesConfig({ activeSessionIdRef: { current: null } }))

    await act(async () => {
      await config.result.current.refreshHermesConfig()
    })

    // Mounted the way a session tile's ChatBar mounts it: nobody hands it the cap.
    const recorder = renderHook(() =>
      useVoiceRecorder({ focusInput: () => undefined, onTranscribeAudio: async () => '', onTranscript: () => undefined })
    )

    await act(async () => {
      recorder.result.current.dictate()
      await Promise.resolve()
    })
    expect(micHandle.start).toHaveBeenCalledTimes(1)

    await act(async () => {
      await vi.advanceTimersByTimeAsync(121_000)
    })
    expect(micHandle.stop).not.toHaveBeenCalled()

    await act(async () => {
      await vi.advanceTimersByTimeAsync(480_000)
    })
    expect(micHandle.stop).toHaveBeenCalledTimes(1)
  })
})
