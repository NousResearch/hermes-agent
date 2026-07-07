import { act, renderHook, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { useVoiceConversation } from './use-voice-conversation'

interface StartOptions {
  onSilence: () => void
}

const mic = vi.hoisted(() => ({
  startOptions: null as StartOptions | null,
  handle: {
    cancel: vi.fn(),
    start: vi.fn(async (options: StartOptions) => {
      mic.startOptions = options
    }),
    stop: vi.fn(async () => ({ audio: new Blob(['audio']), heardSpeech: true }))
  }
}))

vi.mock('@/lib/voice-playback', () => ({ playSpeechText: vi.fn(), stopVoicePlayback: vi.fn() }))
vi.mock('@/store/notifications', () => ({ notify: vi.fn(), notifyError: vi.fn() }))
vi.mock('./use-mic-recorder', () => ({
  useMicRecorder: vi.fn(() => ({ handle: mic.handle, level: 0 }))
}))

describe('useVoiceConversation transcript hold gate', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mic.startOptions = null
  })

  it('holds a risky transcript before it reaches onSubmit', async () => {
    const onSubmit = vi.fn(async () => {})
    const onTranscriptHeld = vi.fn()

    const props = {
      busy: false,
      consumePendingResponse: vi.fn(),
      enabled: false,
      onSubmit,
      onTranscriptHeld,
      onTranscribeAudio: vi.fn(async () => 'delete all the files in downloads'),
      pendingResponse: vi.fn(() => null),
      shouldHoldTranscript: vi.fn(() => true)
    }

    const view = renderHook(({ enabled }) => useVoiceConversation({ ...props, enabled }), {
      initialProps: { enabled: false }
    })

    view.rerender({ enabled: true })

    await waitFor(() => expect(mic.handle.start).toHaveBeenCalled())
    await act(async () => mic.startOptions!.onSilence())

    await waitFor(() => expect(onTranscriptHeld).toHaveBeenCalledWith('delete all the files in downloads'))
    expect(onSubmit).not.toHaveBeenCalled()
  })
})
