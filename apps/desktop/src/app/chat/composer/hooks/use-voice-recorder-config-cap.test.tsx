import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { setApiRequestConnection, setApiRequestProfile } from '@/api/client'
import { useHermesConfig } from '@/app/session/hooks/use-hermes-config'
import { getHermesConfig } from '@/hermes'

import type { MicRecording } from './use-mic-recorder'
import { useVoiceRecorder } from './use-voice-recorder'

// #154537: tiles must honour their own config, including Bot tiles that stay
// visible while the foreground profile (or connection) changes.
let recording = false
let scope: { connectionId?: string; profile?: string } = {}
const api = vi.fn()

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

vi.mock('../scope', () => ({ useComposerScope: () => scope }))

vi.mock('@/hermes', async () => {
  const { resolveOwnerNow } = await import('@/api/client')

  return {
    getHermesConfig: vi.fn(),
    getHermesConfigDefaults: vi.fn().mockResolvedValue({}),
    resolveOwnerNow,
    setApiRequestProfile: vi.fn()
  }
})

describe('dictation cap from voice.max_recording_seconds', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    recording = false
    scope = {}
    setApiRequestConnection('gateway-a')
    setApiRequestProfile('default')
    micHandle.start.mockClear()
    micHandle.stop.mockClear()
    api.mockReset()
    Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { api } })
  })

  afterEach(() => {
    cleanup()
    setApiRequestConnection(null)
    setApiRequestProfile(null)
    Reflect.deleteProperty(window, 'hermesDesktop')
    vi.useRealTimers()
  })

  async function startRecorder(foregroundCap: number) {
    vi.mocked(getHermesConfig).mockResolvedValue({ voice: { max_recording_seconds: foregroundCap } } as never)
    const config = renderHook(() => useHermesConfig({ activeSessionIdRef: { current: null } }))

    await act(async () => {
      await config.result.current.refreshHermesConfig()
    })

    const recorder = renderHook(() =>
      useVoiceRecorder({
        focusInput: () => undefined,
        onTranscribeAudio: async () => '',
        onTranscript: () => undefined
      })
    )

    await act(async () => {
      recorder.result.current.dictate()
    })
    expect(micHandle.start).toHaveBeenCalledTimes(1)

    return recorder
  }

  async function expectCap(seconds: number) {
    await act(async () => {
      await vi.advanceTimersByTimeAsync(seconds * 1000 - 1)
    })
    expect(micHandle.stop).not.toHaveBeenCalled()

    await act(async () => {
      await vi.advanceTimersByTimeAsync(1)
    })
    expect(micHandle.stop).toHaveBeenCalledTimes(1)
  }

  it('a composer without an explicit owner uses the ambient owner at mic-open', async () => {
    api.mockResolvedValue({ voice: { max_recording_seconds: 600 } })
    await startRecorder(600)
    await expectCap(600)
    expect(api).toHaveBeenCalledWith(
      expect.objectContaining({
        connectionId: 'gateway-a',
        path: '/api/config',
        profile: 'default'
      })
    )
  })

  it.each([
    { foreground: 600, owner: 120 },
    { foreground: 120, owner: 600 }
  ])('uses the tile owner cap ($owner), not the foreground cap ($foreground)', async ({ foreground, owner }) => {
    scope = { connectionId: 'bot-gateway', profile: 'bot-owner' }
    api.mockImplementation(async (request: { connectionId?: string; profile?: string }) => ({
      voice: {
        max_recording_seconds:
          request.connectionId === 'bot-gateway' && request.profile === 'bot-owner' ? owner : foreground
      }
    }))
    const recorder = await startRecorder(foreground)

    // Neither a foreground switch nor a scope rerender can retarget this take.
    setApiRequestConnection('gateway-b')
    setApiRequestProfile('other-profile')
    scope = { connectionId: 'gateway-b', profile: 'other-profile' }
    recorder.rerender()
    await expectCap(owner)
    expect(api).toHaveBeenCalledWith(
      expect.objectContaining({
        connectionId: 'bot-gateway',
        path: '/api/config',
        profile: 'bot-owner'
      })
    )
  })

  it.each([
    { value: undefined, seconds: 120 },
    { value: false, seconds: 120 },
    { value: 0, seconds: 120 },
    { value: Number.NaN, seconds: 120 },
    { value: 900, seconds: 600 },
    { value: 1.9, seconds: 1 }
  ])('defaults or clamps an owner cap of $value to $seconds seconds', async ({ value, seconds }) => {
    api.mockResolvedValue({ voice: { max_recording_seconds: value } })
    await startRecorder(600)
    await expectCap(seconds)
  })

  it('opens the mic without waiting for the owner config, then applies its cap', async () => {
    const config = Promise.withResolvers<Record<string, unknown>>()

    api.mockReturnValue(config.promise)
    await startRecorder(600)

    // The owner has not answered 30 s in: the take is already recording.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(30_000)
    })
    await act(async () => {
      config.resolve({ voice: { max_recording_seconds: 600 } })
    })

    // The cap counts from mic-open, not from the late answer.
    await expectCap(570)
  })

  it('a late owner config does not restart a take that already stopped', async () => {
    const config = Promise.withResolvers<Record<string, unknown>>()

    api.mockReturnValue(config.promise)
    const recorder = await startRecorder(600)

    await act(async () => {
      recorder.result.current.dictate()
    })
    expect(micHandle.stop).toHaveBeenCalledTimes(1)

    await act(async () => {
      config.resolve({ voice: { max_recording_seconds: 600 } })
    })
    await act(async () => {
      await vi.advanceTimersByTimeAsync(601_000)
    })
    expect(micHandle.stop).toHaveBeenCalledTimes(1)
  })

  it('uses the default when the owner config is unavailable', async () => {
    api.mockRejectedValue(new Error('offline'))
    await startRecorder(600)
    await expectCap(120)
  })
})
