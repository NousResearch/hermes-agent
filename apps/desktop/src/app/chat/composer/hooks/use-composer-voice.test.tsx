import { act, cleanup, renderHook } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, expect, it, vi } from 'vitest'

const { handle, request } = vi.hoisted(() => ({
  handle: { start: vi.fn(async () => undefined), cancel: vi.fn(), stop: vi.fn() },
  request: vi.fn()
}))

vi.mock('./use-mic-recorder', () => ({ useMicRecorder: () => ({ handle, level: 0 }) }))
vi.mock('./use-auto-speak-replies', () => ({ useAutoSpeakReplies: vi.fn() }))
vi.mock('./use-voice-recorder', () => ({ useVoiceRecorder: () => ({ voiceStatus: 'idle' }) }))
vi.mock('../scope', () => ({ useComposerScope: () => ({ $messages: { get: () => [] } }) }))
vi.mock('@/store/gateway', () => ({ $gateway: { get: () => ({ request }) } }))
vi.mock('@/store/wake-word', () => ({ resumeWakeAfterVoice: vi.fn(async () => undefined) }))
vi.mock('@/store/voice-prefs', () => ({
  $autoSpeakReplies: atom(false),
  $voiceStopPhrase: atom(null),
  setAutoSpeakReplies: vi.fn()
}))
vi.mock('@/lib/tts-lease', () => ({
  CONVERSATION_LEASE: 'conversation',
  READ_ALOUD_LEASE: 'read-aloud',
  syncTtsLease: vi.fn(async () => undefined)
}))
vi.mock('@/lib/voice-playback', () => ({ stopVoicePlayback: vi.fn() }))
vi.mock('@/lib/thinking-sound', () => ({ startThinkingSound: vi.fn(), stopThinkingSound: vi.fn() }))
vi.mock('@/store/notifications', () => ({ notify: vi.fn(), notifyError: vi.fn() }))

import { useComposerVoice } from './use-composer-voice'

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

it('releases the wake-word microphone before opening conversation capture', async () => {
  let releaseWakeMicrophone!: () => void
  request.mockImplementation(() => new Promise<void>(resolve => (releaseWakeMicrophone = resolve)))

  const { result } = renderHook(() =>
    useComposerVoice({
      busy: false,
      clearDraft: vi.fn(),
      disabled: false,
      focusInput: vi.fn(),
      insertText: vi.fn(),
      maxRecordingSeconds: 60,
      onSubmit: vi.fn(),
      onTranscribeAudio: vi.fn(async () => 'Hello Hermes'),
      sessionId: null,
      target: 'main'
    })
  )

  await act(async () => result.current.startConversation())
  expect(request).toHaveBeenCalledExactlyOnceWith('wake.pause', {})
  expect(handle.start).not.toHaveBeenCalled()

  await act(async () => releaseWakeMicrophone())
  expect(handle.start).toHaveBeenCalledOnce()
  expect(result.current.conversation.status).toBe('listening')
})
