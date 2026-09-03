import { act, renderHook } from '@testing-library/react'
import { beforeEach, expect, test, vi } from 'vitest'

const mocks = await vi.hoisted(async () => {
  const { atom } = await import('nanostores')
  const messages = atom<unknown[]>([])

  const useVoiceConversation = vi.fn(() => ({
    end: vi.fn(async () => undefined),
    level: 0,
    muted: false,
    start: vi.fn(async () => undefined),
    status: 'idle' as const,
    stopTurn: vi.fn(),
    toggleMute: vi.fn()
  }))

  return { messages, useVoiceConversation }
})

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      assistant: { thread: { readAloudFailed: 'read aloud failed' } },
      notifications: { voice: { sayStopToEnd: (phrase: string) => phrase } },
      settings: { config: { autosaveFailed: 'autosave failed' } }
    }
  })
}))
vi.mock('../scope', () => ({ useComposerScope: () => ({ $messages: mocks.messages }) }))
vi.mock('./use-voice-conversation', () => ({ useVoiceConversation: mocks.useVoiceConversation }))
vi.mock('./use-voice-recorder', () => ({
  useVoiceRecorder: () => ({ dictate: vi.fn(), voiceActivityState: null, voiceStatus: 'idle' })
}))
vi.mock('./use-auto-speak-replies', () => ({ useAutoSpeakReplies: vi.fn() }))
vi.mock('@/lib/haptics', () => ({ triggerHaptic: vi.fn() }))
vi.mock('@/lib/wake-indicator', () => ({
  clearWakeIndicator: vi.fn(),
  syncWakeIndicatorWithVoice: vi.fn(() => false)
}))
vi.mock('@/store/ambient', () => ({ ownsAmbientCue: vi.fn(async () => true) }))
vi.mock('@/store/notifications', () => ({ notify: vi.fn(), notifyError: vi.fn() }))
vi.mock('@/store/wake-word', () => ({
  resumeWakeAfterVoice: vi.fn(async () => undefined),
  stopClientCapture: vi.fn(async () => undefined)
}))

import { useComposerVoice } from './use-composer-voice'

beforeEach(() => {
  mocks.messages.set([])
  mocks.useVoiceConversation.mockClear()
})

test('re-renders once when an in-progress assistant reply first becomes speakable', () => {
  renderHook(() =>
    useComposerVoice({
      busy: true,
      clearDraft: vi.fn(),
      disabled: false,
      focusInput: vi.fn(),
      insertText: vi.fn(),
      maxRecordingSeconds: 60,
      onSubmit: vi.fn(async () => true),
      onTranscribeAudio: vi.fn(async () => ''),
      sessionId: 'voice-session',
      target: 'main'
    })
  )

  const callsBeforeDelta = mocks.useVoiceConversation.mock.calls.length

  act(() => {
    mocks.messages.set([
      {
        id: 'assistant-stream',
        pending: true,
        role: 'assistant',
        parts: [{ type: 'text', text: 'Natural selection is' }]
      }
    ])
  })

  const callsAfterFirstDelta = mocks.useVoiceConversation.mock.calls.length
  expect(callsAfterFirstDelta).toBeGreaterThan(callsBeforeDelta)

  act(() => {
    mocks.messages.set([
      {
        id: 'assistant-stream',
        pending: true,
        role: 'assistant',
        parts: [{ type: 'text', text: 'Natural selection is the differential survival.' }]
      }
    ])
  })

  expect(mocks.useVoiceConversation).toHaveBeenCalledTimes(callsAfterFirstDelta)
})
