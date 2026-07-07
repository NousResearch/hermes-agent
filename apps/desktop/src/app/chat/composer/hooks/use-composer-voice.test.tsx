import { act, renderHook } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { notify } from '@/store/notifications'

import { useComposerVoice } from './use-composer-voice'

interface CapturedConversationOptions {
  enabled: boolean
  onSubmit: (text: string) => Promise<void> | void
  shouldHoldTranscript?: (text: string) => boolean
  onTranscriptHeld?: (text: string) => void
}

const captured = vi.hoisted(() => ({ options: null as CapturedConversationOptions | null }))

vi.mock('@/lib/haptics', () => ({ triggerHaptic: vi.fn() }))
vi.mock('@/store/composer-input-history', () => ({ resetBrowseState: vi.fn() }))
vi.mock('@/store/notifications', () => ({ notify: vi.fn(() => 'id'), notifyError: vi.fn(() => 'id') }))
vi.mock('@/store/session', async () => {
  const { atom } = await import('nanostores')

  return { $messages: atom([]) }
})
vi.mock('@/store/voice-prefs', async () => {
  const { atom } = await import('nanostores')

  return { $autoSpeakReplies: atom(false), setAutoSpeakReplies: vi.fn(async () => {}) }
})
vi.mock('../focus', () => ({ onComposerVoiceToggleRequest: vi.fn(() => () => {}) }))
vi.mock('./use-auto-speak-replies', () => ({ useAutoSpeakReplies: vi.fn() }))
vi.mock('./use-voice-recorder', () => ({
  useVoiceRecorder: vi.fn(() => ({
    dictate: vi.fn(),
    voiceActivityState: { elapsedSeconds: 0, level: 0, status: 'idle' },
    voiceStatus: 'idle'
  }))
}))
vi.mock('./use-voice-conversation', () => ({
  useVoiceConversation: vi.fn((options: CapturedConversationOptions) => {
    captured.options = options

    return {
      end: vi.fn(async () => {}),
      level: 0,
      muted: false,
      start: vi.fn(async () => {}),
      status: 'idle',
      stopTurn: vi.fn(),
      toggleMute: vi.fn()
    }
  })
}))

function renderComposerVoice() {
  const props = {
    busy: false,
    clearDraft: vi.fn(),
    disabled: false,
    focusInput: vi.fn(),
    insertText: vi.fn(),
    maxRecordingSeconds: 60,
    onSubmit: vi.fn(async () => true),
    onTranscribeAudio: vi.fn(async () => ''),
    sessionId: 'session-1'
  }

  const view = renderHook(() => useComposerVoice(props))

  return { props, ...view }
}

describe('useComposerVoice hands-free confirmation gate', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    captured.options = null
  })

  it('auto-submits non-risky transcripts as before', async () => {
    const { props } = renderComposerVoice()

    await act(async () => captured.options!.onSubmit("what's the weather like today"))

    expect(props.onSubmit).toHaveBeenCalledWith("what's the weather like today")
    expect(props.clearDraft).toHaveBeenCalled()
    expect(props.insertText).not.toHaveBeenCalled()
    expect(notify).not.toHaveBeenCalled()
  })

  it('holds risky transcripts in the draft instead of submitting them', async () => {
    const { props, result } = renderComposerVoice()

    act(() => result.current.startConversation())
    expect(captured.options!.enabled).toBe(true)

    const transcript = 'delete all the files in the downloads folder'
    expect(captured.options!.shouldHoldTranscript?.(transcript)).toBe(true)
    act(() => captured.options!.onTranscriptHeld?.(transcript))

    expect(props.onSubmit).not.toHaveBeenCalled()
    expect(props.clearDraft).not.toHaveBeenCalled()
    expect(props.insertText).toHaveBeenCalledWith(transcript)
    expect(notify).toHaveBeenCalledWith(expect.objectContaining({ kind: 'warning' }))
    // The hands-free loop deactivates so it cannot keep listening past the warning.
    expect(captured.options!.enabled).toBe(false)
  })
})
