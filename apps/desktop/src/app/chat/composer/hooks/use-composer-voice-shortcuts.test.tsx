import { act, cleanup, render } from '@testing-library/react'
import type { ReactNode } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { markActiveComposer, requestComposerDictation, requestVoiceToggle } from '../focus'
import { ComposerScopeProvider, ComposerSurfaceProvider, MAIN_COMPOSER_SCOPE } from '../scope'

import { useComposerVoice } from './use-composer-voice'

const mocks = vi.hoisted(() => ({
  conversationEnabled: [] as boolean[],
  dictate: vi.fn(),
  endConversation: vi.fn(async () => undefined)
}))

vi.mock('./use-voice-recorder', () => ({
  useVoiceRecorder: () => ({
    dictate: mocks.dictate,
    voiceActivityState: { elapsedSeconds: 0, level: 0, status: 'idle' },
    voiceStatus: 'idle'
  })
}))

vi.mock('./use-voice-conversation', () => ({
  useVoiceConversation: ({ enabled }: { enabled: boolean }) => {
    mocks.conversationEnabled.push(enabled)

    return { end: mocks.endConversation, start: vi.fn(), status: 'idle' }
  }
}))

vi.mock('./use-voice-live-conversation', () => ({
  useVoiceLiveConversation: () => ({ end: mocks.endConversation, start: vi.fn(), status: 'idle' })
}))

vi.mock('./use-auto-speak-replies', () => ({ useAutoSpeakReplies: vi.fn() }))

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      notifications: { voice: {} },
      assistant: { thread: { readAloudFailed: '' } },
      settings: { config: { autosaveFailed: '' } }
    }
  })
}))

vi.mock('@/lib/haptics', () => ({ triggerHaptic: vi.fn() }))
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
vi.mock('@/lib/wake-indicator', () => ({ clearWakeIndicator: vi.fn(), syncWakeIndicatorWithVoice: vi.fn() }))
vi.mock('@/lib/voice-live', () => ({ toLiveHistory: vi.fn(() => []) }))
vi.mock('@/store/notifications', () => ({ notify: vi.fn(), notifyError: vi.fn() }))
vi.mock('@/store/voice-live', async () => {
  const { atom } = await import('nanostores')

  return {
    $voiceLiveStatus: atom(null),
    refreshVoiceLiveStatus: vi.fn(async () => undefined),
    selectedVoiceChatMode: vi.fn(() => 'chained')
  }
})
vi.mock('@/store/voice-prefs', async () => {
  const { atom } = await import('nanostores')

  return {
    $autoSpeakReplies: atom(false),
    $voiceStopPhrase: atom(null),
    setAutoSpeakReplies: vi.fn(async () => undefined)
  }
})
vi.mock('@/store/gateway', async () => {
  const { atom } = await import('nanostores')

  return { $gateway: atom(null) }
})
vi.mock('@/store/composer-input-history', () => ({ resetBrowseState: vi.fn() }))
vi.mock('@/store/wake-word', () => ({ resumeWakeAfterVoice: vi.fn(async () => undefined) }))
vi.mock('../floating-target', () => ({ pinFloatingComposerCapture: vi.fn(() => undefined) }))

function Composer({ disabled, target }: { disabled: boolean; target: string }) {
  useComposerVoice({
    busy: false,
    clearDraft: vi.fn(),
    disabled,
    focusInput: vi.fn(),
    insertText: vi.fn(),
    maxRecordingSeconds: 60,
    onSubmit: vi.fn(async () => true),
    onTranscribeAudio: vi.fn(async () => 'spoken text'),
    sessionId: null,
    target
  })

  return null
}

function mountComposer(target: string, disabled: boolean, hidden = false) {
  const scope = { ...MAIN_COMPOSER_SCOPE, target }

  return (
    <ComposerScopeProvider value={scope}>
      <ComposerSurfaceProvider value={`${target}-surface`}>
        <div data-composer-target={target} data-pane-hidden={hidden ? '' : undefined}>
          <Composer disabled={disabled} target={target} />
        </div>
      </ComposerSurfaceProvider>
    </ComposerScopeProvider>
  )
}

function renderComposers(children: ReactNode) {
  return render(children)
}

afterEach(() => {
  cleanup()
  document.body.innerHTML = ''
  mocks.dictate.mockClear()
  mocks.endConversation.mockClear()
  mocks.conversationEnabled.length = 0
  markActiveComposer('main')
})

describe('composer voice shortcuts', () => {
  it('dictates only on the active visible target and ignores a disabled target', async () => {
    renderComposers(
      <>
        {mountComposer('main', true, true)}
        {mountComposer('tile:front', false)}
      </>
    )
    markActiveComposer('tile:front')

    await act(async () => {
      requestComposerDictation('active')
      await new Promise(resolve => window.setTimeout(resolve, 0))
    })
    expect(mocks.dictate).toHaveBeenCalledTimes(1)

    await act(async () => {
      requestComposerDictation('main')
      await new Promise(resolve => window.setTimeout(resolve, 0))
    })
    expect(mocks.dictate).toHaveBeenCalledTimes(1)
  })

  it('forwards repeated dictation requests without toggling voice conversation', async () => {
    renderComposers(mountComposer('main', false))

    await act(async () => {
      requestComposerDictation('active')
      requestComposerDictation('active')
      await new Promise(resolve => window.setTimeout(resolve, 0))
    })
    expect(mocks.dictate).toHaveBeenCalledTimes(2)
    expect(mocks.endConversation).not.toHaveBeenCalled()

    await act(async () => {
      requestVoiceToggle('active')
      await new Promise(resolve => window.setTimeout(resolve, 0))
    })
    expect(mocks.dictate).toHaveBeenCalledTimes(2)
    expect(mocks.conversationEnabled).toContain(true)
  })
})
