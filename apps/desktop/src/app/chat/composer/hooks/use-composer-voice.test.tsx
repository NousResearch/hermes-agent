import { act, cleanup, renderHook } from '@testing-library/react'
import type * as Nanostores from 'nanostores'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { useComposerVoice } from './use-composer-voice'

const mocks = vi.hoisted(() => ({
  clearDraft: vi.fn(),
  conversationSubmits: [] as Array<(text: string) => Promise<void>>,
  onSubmit: vi.fn(async () => true),
  onInterrupt: vi.fn(async () => undefined),
  endConversation: vi.fn(async () => undefined),
  notifyError: vi.fn(),
  resetBrowseState: vi.fn(),
  triggerHaptic: vi.fn()
}))

vi.mock('@nanostores/react', () => ({
  useStore: (store: unknown) => store
}))

vi.mock('nanostores', async () => {
  const actual = await vi.importActual<typeof Nanostores>('nanostores')

  return { ...actual, computed: (deps: unknown, fn: unknown) => fn }
})

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      common: { error: 'Error', retry: 'Retry' },
      assistant: { thread: { readAloudFailed: '' } },
      notifications: { voice: { sayStopToEnd: () => '' } },
      settings: { config: { autosaveFailed: '' } }
    }
  })
}))

vi.mock('@/lib/haptics', () => ({
  triggerHaptic: mocks.triggerHaptic
}))

vi.mock('@/lib/spoken-reply', () => ({
  adoptSpokenReplySession: vi.fn(),
  markAssistantIdSpoken: vi.fn(),
  resolveSpokenReply: vi.fn(() => null)
}))

vi.mock('@/lib/tts-lease', () => ({
  CONVERSATION_LEASE: 'conversation',
  READ_ALOUD_LEASE: 'read-aloud',
  syncTtsLease: vi.fn(async () => undefined)
}))

vi.mock('@/lib/wake-indicator', () => ({
  clearWakeIndicator: vi.fn(),
  syncWakeIndicatorWithVoice: vi.fn(() => false)
}))

vi.mock('@/store/composer', () => ({
  $voiceConversationStartRequest: { get: () => null },
  takeVoiceConversationStart: vi.fn(() => false)
}))

vi.mock('@/store/composer-input-history', () => ({
  resetBrowseState: mocks.resetBrowseState
}))

vi.mock('@/store/gateway', () => ({
  $gateway: { get: () => null }
}))

vi.mock('@/store/notifications', () => ({
  notify: vi.fn(),
  notifyError: mocks.notifyError
}))

vi.mock('@/store/voice-prefs', () => ({
  $autoSpeakReplies: { get: () => false },
  $voiceStopPhrase: { get: () => null },
  setAutoSpeakReplies: vi.fn(async () => undefined)
}))

vi.mock('@/store/wake-word', () => ({
  resumeWakeAfterVoice: vi.fn(async () => undefined)
}))

vi.mock('../focus', () => ({
  onComposerDictationRequest: vi.fn(() => () => undefined),
  onComposerVoiceToggleRequest: vi.fn(() => () => undefined)
}))

vi.mock('../scope', () => ({
  useComposerScope: () => ({ $messages: { get: () => [] } }),
  useComposerSurfaceId: () => null
}))

vi.mock('./use-auto-speak-replies', () => ({
  useAutoSpeakReplies: vi.fn()
}))

vi.mock('./use-voice-conversation', () => ({
  useVoiceConversation: ({ onSubmit }: { onSubmit: (text: string) => Promise<void> }) => {
    mocks.conversationSubmits.push(onSubmit)

    return {
      end: mocks.endConversation,
      level: 0,
      muted: false,
      start: vi.fn(async () => undefined),
      status: 'idle',
      stopTurn: vi.fn(),
      toggleMute: vi.fn()
    }
  }
}))

vi.mock('./use-voice-recorder', () => ({
  useVoiceRecorder: () => ({
    dictate: vi.fn(),
    voiceActivityState: { elapsedSeconds: 0, level: 0, status: 'idle' },
    voiceStatus: 'idle'
  })
}))

function renderVoice() {
  return renderHook(
    ({ sessionId, disabled }) =>
      useComposerVoice({
        busy: false,
        clearDraft: mocks.clearDraft,
        disabled,
        focusInput: vi.fn(),
        insertText: vi.fn(),
        maxRecordingSeconds: 60,
        onInterrupt: mocks.onInterrupt,
        onSubmit: mocks.onSubmit,
        onTranscribeAudio: vi.fn(async () => ''),
        sessionId,
        target: 'tile:test'
      }),
    { initialProps: { sessionId: 'session-1', disabled: false } }
  )
}

describe('useComposerVoice voice submission', () => {
  afterEach(() => {
    cleanup()
    vi.clearAllMocks()
    mocks.conversationSubmits.length = 0
  })

  it('uses live busy state when a retained voice callback submits after barge-in', async () => {
    const hook = renderHook(
      ({ busy }) =>
        useComposerVoice({
          busy,
          clearDraft: mocks.clearDraft,
          disabled: false,
          focusInput: vi.fn(),
          insertText: vi.fn(),
          maxRecordingSeconds: 60,
          onSubmit: mocks.onSubmit,
          onTranscribeAudio: vi.fn(async () => ''),
          sessionId: 'session-1',
          target: 'tile:test'
        }),
      { initialProps: { busy: true } }
    )

    const retainedSubmit = mocks.conversationSubmits[0]

    hook.rerender({ busy: false })
    await act(async () => retainedSubmit('change direction'))

    expect(mocks.onSubmit).toHaveBeenCalledWith('change direction', { voiceTurn: true })
    expect(mocks.clearDraft).toHaveBeenCalledTimes(1)
    expect(mocks.resetBrowseState).toHaveBeenCalledWith('session-1')
    expect(mocks.triggerHaptic).toHaveBeenCalledWith('submit')

    hook.rerender({ busy: true })
    await act(async () => retainedSubmit('do not double-submit'))

    expect(mocks.onSubmit).toHaveBeenCalledTimes(1)
    expect(mocks.clearDraft).toHaveBeenCalledTimes(1)
  })

  it.each(['submit', 'interrupt'] as const)(
    'reports rejected plugin %s callbacks without an unhandled rejection',
    async action => {
      const failure = new Error('callback failed')

      if (action === 'submit') {
        mocks.onSubmit.mockRejectedValueOnce(failure)
      } else {
        mocks.onInterrupt.mockRejectedValueOnce(failure)
      }

      const hook = renderVoice()
      await act(async () => {
        const accepted =
          action === 'submit'
            ? hook.result.current.voiceController.submitText('hello')
            : hook.result.current.voiceController.interrupt()

        expect(accepted).toBe(true)
      })
      expect(mocks.notifyError).toHaveBeenCalledWith(failure, 'Error')
    }
  )

  it('rejects retained plugin controllers after navigation, disable and unmount', () => {
    const hook = renderVoice()
    const first = hook.result.current.voiceController
    hook.rerender({ sessionId: 'session-2', disabled: false })
    expect(first.submitText('stale')).toBe(false)
    expect(first.interrupt()).toBe(false)
    const second = hook.result.current.voiceController
    hook.rerender({ sessionId: 'session-2', disabled: true })
    expect(second.submitText('disabled')).toBe(false)
    hook.rerender({ sessionId: 'session-2', disabled: false })
    const third = hook.result.current.voiceController
    hook.unmount()
    expect(third.submitText('unmounted')).toBe(false)
    expect(mocks.onSubmit).not.toHaveBeenCalled()
    expect(mocks.onInterrupt).not.toHaveBeenCalled()
  })

  it('keeps retained plugin controllers from acquiring the mic during native voice', async () => {
    const hook = renderVoice()
    const controller = hook.result.current.voiceController
    act(() => hook.result.current.startConversation())
    await expect(controller.acquire()).resolves.toBeNull()
  })

  it('reserves native capture across composers until the native engine has stopped', async () => {
    let stopped!: () => void

    const stopping = new Promise<void>(resolve => {
      stopped = resolve
    })

    mocks.endConversation.mockImplementation(() => stopping)
    const native = renderVoice()
    const plugin = renderVoice()
    act(() => native.result.current.startConversation())
    await expect(plugin.result.current.voiceController.acquire()).resolves.toBeNull()
    act(() => native.result.current.endConversation())
    await act(async () => {
      await Promise.resolve()
    })
    await expect(plugin.result.current.voiceController.acquire()).resolves.toBeNull()
    await act(async () => {
      stopped()
    })
    const lease = await plugin.result.current.voiceController.acquire()
    expect(lease).not.toBeNull()
    lease?.release()
    mocks.endConversation.mockResolvedValue(undefined)
  })

  it('offers retry when native teardown fails without releasing a potentially live mic', async () => {
    const failure = new Error('teardown failed')
    mocks.endConversation.mockRejectedValueOnce(failure)
    const native = renderVoice()
    const plugin = renderVoice()
    act(() => native.result.current.startConversation())
    await act(async () => native.unmount())
    await expect(plugin.result.current.voiceController.acquire()).resolves.toBeNull()
    const retry = mocks.notifyError.mock.calls.find(call => call[0] === failure)?.[2]?.action
    expect(retry?.label).toBe('Retry')
    await act(async () => retry.onClick())
    const lease = await plugin.result.current.voiceController.acquire()
    expect(lease).not.toBeNull()
    lease?.release()
  })
})
