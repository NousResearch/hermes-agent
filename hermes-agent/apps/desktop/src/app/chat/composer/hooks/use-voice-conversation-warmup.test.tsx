import { act, cleanup, renderHook, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { MicRecording } from './use-mic-recorder'
import { useVoiceConversation } from './use-voice-conversation'

// #105955 review: the warm-up fired when the conversation starts listening
// is a READINESS BARRIER before transcription — handleTurn() awaits it before
// handing the clip to `onTranscribeAudio`, so a cold local model load settles
// inside the lease request's own budget instead of eating the transcription
// request's decode deadline (190 s cold load + 3 s clip vs a 180 s floor).

const micHandle = {
  cancel: vi.fn(),
  start: vi.fn(async () => undefined),
  stop: vi.fn<() => Promise<MicRecording | null>>(async () => null)
}

vi.mock('./use-mic-recorder', () => ({
  useMicRecorder: () => ({ handle: micHandle, level: 0, recording: false })
}))

const syncSttLeaseSpy = vi.fn(async (..._args: unknown[]): Promise<void> => undefined)

vi.mock('@/lib/stt-lease', () => ({
  syncSttLease: (...args: unknown[]) => syncSttLeaseSpy(...args),
  VOICE_INPUT_LEASE: 'desktop:voice-input:test'
}))

vi.mock('@/lib/voice-barge-in', () => ({
  monitorSpeechDuringPlayback: () => () => undefined
}))

vi.mock('@/lib/voice-playback', () => ({
  markVoicePlaybackInterrupted: vi.fn(),
  playSpeechText: vi.fn(async () => true),
  startSpeechStream: vi.fn(async () => null),
  stopVoicePlayback: vi.fn(),
  takeVoicePlaybackInterrupted: vi.fn(() => true)
}))

vi.mock('@/lib/thinking-sound', () => ({
  startThinkingSound: vi.fn(),
  stopThinkingSound: vi.fn()
}))

vi.mock('@/lib/speech-text', () => ({
  IncrementalSpeechSentenceBuffer: class {
    append() {
      return []
    }
    flush() {
      return []
    }
  }
}))

vi.mock('@/lib/voice-stop-word', () => ({ isVoiceStopCommand: () => false }))
vi.mock('@/lib/voice-tts-echo', () => ({ isTtsEcho: () => false }))

vi.mock('@/store/notifications', () => ({
  notify: vi.fn(),
  notifyError: vi.fn()
}))

vi.mock('@/store/voice-playback', async () => {
  const { atom } = await import('nanostores')

  return { $voicePlayback: atom({ sequence: 0, status: 'idle' }) }
})

vi.mock('@/store/voice-prefs', async () => {
  const { atom } = await import('nanostores')

  return {
    $autoSpeakReplies: atom(false),
    $bargeInEnabled: atom(true),
    $bargeInThresholdMultiplier: atom(1),
    $voiceSilenceMs: atom(1250)
  }
})

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      notifications: {
        voice: {
          configureSpeechToText: 'configure STT',
          couldNotStartSession: 'could not start',
          microphoneFailed: 'mic failed',
          playbackFailed: 'playback failed',
          recordingFailed: 'recording failed',
          transcriptionFailed: 'transcription failed',
          unavailable: 'unavailable'
        }
      }
    }
  })
}))

vi.mock('../scope', () => ({
  useComposerScope: () => ({ $messages: { get: () => [] }, connectionId: undefined, profile: undefined })
}))

// The ambient selection resolveOwnerNow reads; tests flip it mid-conversation.
const ambient = { connectionId: 'gateway-a' as null | string, profile: 'worker_alpha' as null | string }

vi.mock('@/hermes', () => ({
  resolveOwnerNow: (owner?: { connectionId?: string; profile?: string }) => ({
    connectionId: owner?.connectionId || ambient.connectionId,
    profile: owner?.profile || ambient.profile
  })
}))

const OWNER_A = { connectionId: 'gateway-a', profile: 'worker_alpha' }

describe('useVoiceConversation STT readiness barrier', () => {
  beforeEach(() => {
    cleanup()
    ambient.connectionId = 'gateway-a'
    ambient.profile = 'worker_alpha'
    micHandle.start.mockClear()
    micHandle.cancel.mockClear()
    micHandle.stop.mockReset()
    micHandle.stop.mockImplementation(async () => ({ audio: new Blob(), durationMs: 900, heardSpeech: true }))
    syncSttLeaseSpy.mockReset()
    syncSttLeaseSpy.mockImplementation(async () => undefined)
  })

  afterEach(() => {
    cleanup()
  })

  function renderConversation(
    onTranscribeAudio: (audio: Blob, owner?: unknown) => Promise<string>,
    onSubmit: (text: string) => void = () => undefined
  ) {
    const hook = renderHook(() =>
      useVoiceConversation({
        busy: false,
        consumePendingResponse: () => undefined,
        enabled: true,
        onSubmit,
        onTranscribeAudio,
        pendingResponse: () => null
      })
    )

    return hook
  }

  it('waits for the listening warm-up before transcribing the captured utterance', async () => {
    // Cold load in flight: the acquire stays open past the silence stop.
    let settleWarmup!: () => void
    syncSttLeaseSpy.mockImplementation(
      () =>
        new Promise<void>(resolve => {
          settleWarmup = resolve
        })
    )

    const transcribe = vi.fn(async () => 'hello world')
    const hook = renderConversation(transcribe)

    // Listening starts: the mic opens without waiting on the warm-up.
    await act(async () => {
      await hook.result.current.start()
    })
    await act(async () => {
      await Promise.resolve()
    })
    expect(micHandle.start).toHaveBeenCalledTimes(1)
    expect(syncSttLeaseSpy).toHaveBeenCalledWith('desktop:voice-input:test', true, OWNER_A)

    // The VAD silence callback fires a turn while the cold load is in flight.
    let turnPromise: Promise<void> = Promise.resolve()
    await act(async () => {
      micHandle.stop.mockImplementation(async () => ({ audio: new Blob(), durationMs: 900, heardSpeech: true }))
      turnPromise = Promise.resolve(hook.result.current.stopTurn())
      await Promise.resolve()
      await Promise.resolve()
    })

    // The barrier holds: transcription has not started inside the load wait.
    expect(transcribe).not.toHaveBeenCalled()

    settleWarmup()
    await act(async () => {
      await turnPromise
    })

    await waitFor(() => expect(transcribe).toHaveBeenCalledTimes(1))
    expect(await transcribe.mock.results[0].value).toBe('hello world')
  })

  // #128668 review (P1): end() while the warm-up is still settling fences the
  // turn — no transcription, no submit into a conversation that is over.
  it('drops the turn when the conversation ends while the warm-up is settling', async () => {
    let settleWarmup!: () => void
    syncSttLeaseSpy.mockImplementationOnce(
      () =>
        new Promise<void>(resolve => {
          settleWarmup = resolve
        })
    )

    const transcribe = vi.fn(async () => 'hello world')
    const onSubmit = vi.fn()
    const hook = renderConversation(transcribe, onSubmit)

    await act(async () => {
      await hook.result.current.start()
    })

    let turnPromise: Promise<void> = Promise.resolve()
    await act(async () => {
      turnPromise = Promise.resolve(hook.result.current.stopTurn())
      await Promise.resolve()
      await Promise.resolve()
    })

    await act(async () => {
      hook.result.current.end()
    })
    expect(syncSttLeaseSpy).toHaveBeenLastCalledWith('desktop:voice-input:test', false, OWNER_A)

    settleWarmup()
    await act(async () => {
      await turnPromise
    })

    expect(transcribe).not.toHaveBeenCalled()
    expect(onSubmit).not.toHaveBeenCalled()
  })

  // #128668 review: one owner per conversation, resolved at start().
  it('transcribes and releases on the start() owner across a mid-conversation switch', async () => {
    const transcribe = vi.fn(async (_audio: Blob, _owner?: unknown) => 'hello world')
    const hook = renderConversation(transcribe)

    await act(async () => {
      await hook.result.current.start()
    })

    ambient.connectionId = null
    ambient.profile = null

    await act(async () => {
      await hook.result.current.stopTurn()
    })
    await waitFor(() => expect(transcribe).toHaveBeenCalledTimes(1))
    expect(transcribe.mock.calls[0][1]).toEqual(OWNER_A)

    await act(async () => {
      hook.result.current.end()
    })
    expect(syncSttLeaseSpy.mock.calls.every(call => JSON.stringify(call[2]) === JSON.stringify(OWNER_A))).toBe(true)
    expect(syncSttLeaseSpy).toHaveBeenLastCalledWith('desktop:voice-input:test', false, OWNER_A)
  })
})
