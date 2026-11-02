import { act, cleanup, renderHook, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { MicRecording } from './use-mic-recorder'
import { useVoiceConversation } from './use-voice-conversation'

// Regression: after a reply finished playing, the NEXT voice turn must not
// re-speak that reply while the model is thinking. The normal (silence-detected)
// submit path has to mark the previous reply spoken BEFORE submitting — the
// barge path already did — otherwise the turn-drive effect sees `awaiting` plus
// an "unspoken" previous reply and plays it through the whole-text fallback
// until the new answer arrives and barges in.

const playSpeechText = vi.fn(async (_text: string, _options: unknown) => true)

vi.mock('@/lib/voice-barge-in', () => ({
  monitorSpeechDuringPlayback: () => vi.fn()
}))

vi.mock('@/lib/voice-playback', () => ({
  markVoicePlaybackInterrupted: vi.fn(),
  playSpeechText: (text: string, options: unknown) => playSpeechText(text, options),
  startSpeechStream: vi.fn(async () => null),
  stopVoicePlayback: vi.fn()
}))

vi.mock('@/lib/thinking-sound', () => ({
  startThinkingSound: vi.fn(),
  stopThinkingSound: vi.fn()
}))

const micHandle = {
  cancel: vi.fn(),
  start: vi.fn(async () => undefined),
  stop: vi.fn<() => Promise<MicRecording | null>>(async () => null)
}

vi.mock('./use-mic-recorder', () => ({
  useMicRecorder: () => ({ handle: micHandle, level: 0, recording: false })
}))

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      notifications: {
        voice: {
          configureSpeechToText: 'configure STT',
          couldNotStartSession: 'could not start',
          microphoneFailed: 'mic failed',
          playbackFailed: 'playback failed',
          transcriptionFailed: 'transcription failed',
          unavailable: 'unavailable'
        }
      }
    }
  })
}))

vi.mock('@/store/notifications', () => ({
  notify: vi.fn(),
  notifyError: vi.fn()
}))

describe('useVoiceConversation stale reply on next turn', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    micHandle.start.mockResolvedValue(undefined)
    micHandle.stop.mockResolvedValue(null)
  })

  afterEach(cleanup)

  it('marks the previous reply spoken before submitting so it is not re-spoken while thinking', async () => {
    const calls: string[] = []

    // The previous turn's answer still sits at the bottom of the thread and —
    // as in the field — is NOT yet marked spoken when the user's next
    // utterance lands. It stays "pending to speak" until consumed.
    let staleReplyUnspoken = true

    const consumePendingResponse = vi.fn(() => {
      calls.push('consume')
      staleReplyUnspoken = false
    })

    // Mirrors the real app: submitting a turn makes the agent busy.
    const onBusyChange: { current: (busy: boolean) => void } = { current: () => undefined }

    const onSubmit = vi.fn(async () => {
      calls.push('submit')
      onBusyChange.current(true)
    })

    const hook = renderHook(
      ({ busy }: { busy: boolean }) =>
        useVoiceConversation({
          busy,
          consumePendingResponse,
          enabled: true,
          onInterrupt: vi.fn(),
          onStopWord: vi.fn(),
          onSubmit,
          onTranscribeAudio: vi.fn(async () => 'what should we do today'),
          pendingResponse: () => (staleReplyUnspoken ? { id: 'reply-1', pending: false, text: 'old answer' } : null)
        }),
      { initialProps: { busy: false } }
    )

    onBusyChange.current = busy => hook.rerender({ busy })

    await act(async () => {
      await hook.result.current.start()
    })
    await waitFor(() => expect(hook.result.current.status).toBe('listening'))

    // start() consumes once so an already-present reply is never read when the
    // conversation opens; only the submit-time ordering is under test here.
    calls.length = 0

    micHandle.stop.mockResolvedValueOnce({
      audio: new Blob(['q'], { type: 'audio/webm' }),
      durationMs: 900,
      heardSpeech: true
    })

    await act(async () => {
      hook.result.current.stopTurn()
    })
    await waitFor(() => expect(onSubmit).toHaveBeenCalledTimes(1))

    // Consumed strictly before the new prompt goes out (barge-path parity).
    expect(calls).toEqual(['consume', 'submit'])

    // And the stale reply never reaches the whole-text fallback player.
    await waitFor(() => expect(hook.result.current.status).toBe('thinking'))
    expect(playSpeechText).not.toHaveBeenCalled()
  })
})
