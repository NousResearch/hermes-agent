import { act, renderHook, waitFor } from '@testing-library/react'
import { useState } from 'react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { useVoiceConversation } from './use-voice-conversation'

interface PendingVoiceResponse {
  id: string
  pending: boolean
  text: string
}

const recorder = vi.hoisted(() => ({
  cancel: vi.fn(),
  start: vi.fn(),
  stop: vi.fn()
}))

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      notifications: {
        voice: {
          configureSpeechToText: 'Configure speech-to-text',
          couldNotStartSession: 'Could not start voice session',
          microphoneFailed: 'Microphone failed',
          playbackFailed: 'Playback failed',
          transcriptionFailed: 'Transcription failed',
          unavailable: 'Voice unavailable'
        }
      }
    }
  })
}))

vi.mock('@/store/notifications', () => ({
  notify: vi.fn(),
  notifyError: vi.fn()
}))

vi.mock('@/lib/voice-playback', () => ({
  playSpeechText: vi.fn(async () => true),
  prefetchSpeechText: vi.fn(),
  clearSpeechPrefetch: vi.fn(),
  stopVoicePlayback: vi.fn()
}))

vi.mock('./use-mic-recorder', () => ({
  useMicRecorder: () => ({ handle: recorder, level: 0 })
}))

// Drives the hook the way the composer does: `pendingResponse` is a live getter,
// and `busy` is host React state that flips true on submit and false when the
// assistant finishes — so it lands in the same commit as the hook's own
// setStatus('thinking'), as it does in the app.
function setup() {
  const state: { response: PendingVoiceResponse | null } = { response: null }

  const consumePendingResponse = vi.fn(() => {
    state.response = null
  })

  let onSilence: (() => void) | undefined
  recorder.start.mockImplementation(async (options: { onSilence?: () => void }) => {
    onSilence = options?.onSilence
  })
  recorder.stop.mockResolvedValue({
    audio: new Blob(['voice'], { type: 'audio/webm' }),
    durationMs: 1200,
    heardSpeech: true
  })

  const view = renderHook(() => {
    const [busy, setBusy] = useState(false)

    const conversation = useVoiceConversation({
      busy,
      consumePendingResponse,
      enabled: true,
      onSubmit: () => setBusy(true),
      onTranscribeAudio: async () => 'hello',
      pendingResponse: () => state.response
    })

    return { conversation, setBusy }
  })

  const status = () => view.result.current.conversation.status

  // Puts the hook in the post-submit state: awaiting a spoken response, status
  // 'thinking'. Mirrors the real path (start -> listen -> silence -> transcribe
  // -> submit) rather than reaching into refs, so the invariants hold for the
  // transitions the loop actually performs.
  const submitATurn = async () => {
    await act(async () => {
      await view.result.current.conversation.start()
    })
    await waitFor(() => expect(recorder.start).toHaveBeenCalled())

    await act(async () => {
      onSilence?.()
      await Promise.resolve()
      await Promise.resolve()
    })
    await waitFor(() => expect(status()).toBe('thinking'))
    recorder.start.mockClear()
  }

  // The assistant's turn ends: host clears `busy`.
  const finishTurn = async () => {
    await act(async () => view.result.current.setBusy(false))
  }

  return { consumePendingResponse, finishTurn, state, status, submitATurn, view }
}

describe('useVoiceConversation mic re-arm', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    recorder.start.mockResolvedValue(undefined)
  })

  // The bug in #54067: when the reply is fully spoken, `status` is already
  // 'idle' (speak()'s finally set it), so the completion path's setStatus('idle')
  // is a no-op -> React bails out -> no re-render -> the effect never re-runs to
  // reach startListening(). The mic must re-arm from the completion itself, not
  // from an incidental later re-render.
  it('re-arms the mic when a spoken reply completes and status is already idle', async () => {
    const h = setup()
    await h.submitATurn()

    // The assistant streams one sentence, which gets spoken -> status returns to
    // 'idle' via speak()'s finally.
    await act(async () => {
      h.state.response = { id: 'r1', pending: true, text: 'This is a full sentence.' }
    })
    await act(async () => h.view.rerender())
    await waitFor(() => expect(h.status()).toBe('idle'))

    recorder.start.mockClear()

    // The reply completes with nothing further to speak.
    h.state.response = { id: 'r1', pending: false, text: 'This is a full sentence.' }
    await h.finishTurn()

    await waitFor(() => expect(recorder.start).toHaveBeenCalledTimes(1))
  })

  // The sibling early-return path (`!busy && status === 'thinking'`): the turn
  // ends without ever producing a spoken response. The loop must still recover
  // to a listening mic rather than stranding the UI in 'thinking'.
  it('re-arms the mic when a turn ends without a spoken response', async () => {
    const h = setup()
    await h.submitATurn()

    // Turn finishes; no pending response was ever published.
    h.state.response = null
    await h.finishTurn()

    await waitFor(() => expect(recorder.start).toHaveBeenCalledTimes(1))
    await waitFor(() => expect(h.status()).toBe('listening'))
  })

  // The completion path can also be reached while status is still 'thinking' —
  // a reply with no speakable text (tool-only / empty). Same invariant.
  it('re-arms the mic when a reply completes with no speakable text', async () => {
    const h = setup()
    await h.submitATurn()

    h.state.response = { id: 'r1', pending: false, text: '' }
    await h.finishTurn()

    await waitFor(() => expect(recorder.start).toHaveBeenCalledTimes(1))
    await waitFor(() => expect(h.status()).toBe('listening'))
  })
})
