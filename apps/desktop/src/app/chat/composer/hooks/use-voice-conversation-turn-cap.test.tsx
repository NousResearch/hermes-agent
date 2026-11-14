import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $autoSpeakReplies } from '@/store/voice-prefs'

import { useVoiceConversation, voiceTurnCapMs } from './use-voice-conversation'

// A hands-free turn that never goes quiet ends only at the turn ceiling. These
// tests pin that ceiling to `maxTurnSeconds` (voice.max_recording_seconds), so
// continuous speech past a minute is no longer cut and submitted mid-sentence
// (#140482).

const mocks = vi.hoisted(() => ({
  mic: {
    cancel: vi.fn(),
    // Never calls onSilence: the speaker keeps talking for the whole test.
    start: vi.fn(async () => undefined),
    stop: vi.fn(async () => ({
      audio: new Blob(['fixture']),
      heardSpeech: true,
      durationMs: 900
    }))
  }
}))

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  getApiRequestConnection: () => null,
  getApiRequestProfile: () => null,
  hermesApi: vi.fn(async () => ({ ok: true, stt: { mode: 'relay' } })),
  speakText: vi.fn()
}))
vi.mock('@/api/client', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  ownerScoped: (value: unknown) => value ?? {},
  profileScoped: (value: unknown) => value
}))
vi.mock('@/lib/stt-lease', () => ({ syncSttLease: vi.fn(async () => undefined), VOICE_INPUT_LEASE: 'voice-input' }))
vi.mock('./use-mic-recorder', () => ({ useMicRecorder: () => ({ handle: mocks.mic, level: 0 }) }))
vi.mock('@/lib/voice-barge-in', () => ({ monitorSpeechDuringPlayback: () => vi.fn() }))
vi.mock('@/lib/thinking-sound', () => ({ startThinkingSound: vi.fn(), stopThinkingSound: vi.fn() }))
vi.mock('@/store/notifications', () => ({ notify: vi.fn(), notifyError: vi.fn() }))
vi.mock('@/i18n', () => ({ useI18n: () => ({ t: { notifications: { voice: {} } } }) }))

beforeEach(() => {
  $autoSpeakReplies.set(false)
  vi.useFakeTimers()
})

afterEach(() => {
  cleanup()
  vi.useRealTimers()
  vi.clearAllMocks()
})

async function listen(maxTurnSeconds?: number) {
  const onSubmit = vi.fn(async () => undefined)

  const hook = renderHook(() =>
    useVoiceConversation({
      busy: false,
      enabled: true,
      consumePendingResponse: vi.fn(),
      maxTurnSeconds,
      onSubmit,
      onTranscribeAudio: async () => 'a long thought that keeps going',
      pendingResponse: () => null
    })
  )

  await act(async () => {
    await hook.result.current.start()
  })
  expect(mocks.mic.start).toHaveBeenCalledTimes(1)

  const advance = async (ms: number) => {
    await act(async () => {
      await vi.advanceTimersByTimeAsync(ms)
    })
  }

  return { advance, onSubmit }
}

describe('voiceTurnCapMs', () => {
  it('follows the configured seconds, clamped like dictation', () => {
    expect(voiceTurnCapMs(120)).toBe(120_000)
    expect(voiceTurnCapMs(600)).toBe(600_000)
    expect(voiceTurnCapMs(9_999)).toBe(600_000)
    expect(voiceTurnCapMs(0.5)).toBe(1_000)
    expect(voiceTurnCapMs(90.9)).toBe(90_000)
  })

  it('keeps the old 60 s when unset or invalid', () => {
    expect(voiceTurnCapMs(undefined)).toBe(60_000)
    expect(voiceTurnCapMs(0)).toBe(60_000)
    expect(voiceTurnCapMs(-5)).toBe(60_000)
    expect(voiceTurnCapMs(Number.NaN)).toBe(60_000)
  })
})

describe('hands-free turn ceiling', () => {
  it('keeps listening past a minute when the ceiling is raised', async () => {
    const { advance, onSubmit } = await listen(600)

    await advance(61_000)
    expect(mocks.mic.stop).not.toHaveBeenCalled()
    expect(onSubmit).not.toHaveBeenCalled()

    await advance(599_000 - 61_000)
    expect(onSubmit).not.toHaveBeenCalled()

    await advance(2_000)
    expect(mocks.mic.stop).toHaveBeenCalledTimes(1)
    expect(onSubmit).toHaveBeenCalledWith('a long thought that keeps going')
  })

  it('still ends a turn that never goes quiet at the old 60 s by default', async () => {
    const { advance, onSubmit } = await listen()

    await advance(59_000)
    expect(onSubmit).not.toHaveBeenCalled()

    await advance(2_000)
    expect(mocks.mic.stop).toHaveBeenCalledTimes(1)
    expect(onSubmit).toHaveBeenCalledTimes(1)
  })
})
