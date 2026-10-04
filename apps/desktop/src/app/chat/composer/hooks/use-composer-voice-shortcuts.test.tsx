import { act, cleanup, render, screen } from '@testing-library/react'
import type { ReactNode } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { handleWakeVoiceEvent } from '@/app/contrib/wake-voice'
import { group } from '@/components/pane-shell/tree/model'
import { $activeTreeGroup, $layoutTree, activateTreePane } from '@/components/pane-shell/tree/store'
import { $voiceConversationStartSessionId, requestVoiceConversationStart } from '@/store/composer'
import { $activeGatewayProfile, ensureGatewayProfile, newSessionInProfile } from '@/store/profile'
import { $activeSessionId, $selectedStoredSessionId } from '@/store/session'
import { $focusedRuntimeId, $sessionTiles } from '@/store/session-states'

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
vi.mock('@/lib/wake-indicator', () => ({
  activateWakeIndicator: vi.fn(),
  clearWakeIndicator: vi.fn(),
  syncWakeIndicatorWithVoice: vi.fn()
}))
vi.mock('@/lib/wake-sound', () => ({ playWakeSound: vi.fn() }))
vi.mock('@/store/profile', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  ensureGatewayProfile: vi.fn(async () => undefined),
  newSessionInProfile: vi.fn()
}))
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
vi.mock('@/store/wake-word', () => ({ stopClientCapture: vi.fn(), resumeWakeAfterVoice: vi.fn(async () => undefined) }))
vi.mock('../floating-target', () => ({ pinFloatingComposerCapture: vi.fn(() => undefined) }))

function Composer({
  disabled,
  target,
  sessionId = null
}: {
  disabled: boolean
  target: string
  sessionId?: string | null
}) {
  const voice = useComposerVoice({
    busy: false,
    clearDraft: vi.fn(),
    disabled,
    focusInput: vi.fn(),
    insertText: vi.fn(),
    maxRecordingSeconds: 60,
    onSubmit: vi.fn(async () => true),
    onTranscribeAudio: vi.fn(async () => 'spoken text'),
    sessionId,
    target
  })

  return <output data-testid={target}>{String(voice.voiceConversationActive)}</output>
}

function mountComposer(target: string, disabled: boolean, hidden = false, sessionId: string | null = null) {
  const scope = { ...MAIN_COMPOSER_SCOPE, target }

  return (
    <ComposerScopeProvider value={scope}>
      <ComposerSurfaceProvider value={`${target}-surface`}>
        <div data-composer-target={target} data-pane-hidden={hidden ? '' : undefined}>
          <Composer disabled={disabled} sessionId={sessionId} target={target} />
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

describe('wake event routes through runtime focus and the real composer latch', () => {
  function focusThirdTab() {
    $selectedStoredSessionId.set('stored-A')
    $activeSessionId.set('runtime-A')
    $sessionTiles.set([
      { storedSessionId: 'stored-B', runtimeId: 'runtime-B', dir: 'center' },
      { storedSessionId: 'stored-C', runtimeId: 'runtime-C', dir: 'center' }
    ])
    $layoutTree.set(group(['workspace', 'session-tile:stored-B', 'session-tile:stored-C'], { id: 'wake-tabs' }))
    $activeTreeGroup.set('wake-tabs')
    activateTreePane('wake-tabs', 'session-tile:stored-C')
    expect($focusedRuntimeId.get()).toBe('runtime-C')
  }

  it.each([false, true])('explicit profile routing retains precedence (fresh=%s)', async fresh => {
    focusThirdTab()
    $activeGatewayProfile.set('default')
    vi.mocked(ensureGatewayProfile).mockClear()
    vi.mocked(newSessionInProfile).mockClear()
    const actions = { openNewSessionTile: vi.fn(), startFreshSessionDraft: vi.fn() }
    await act(async () => {
      handleWakeVoiceEvent({ type: 'wake.detected', payload: { profile: 'writer', start_new_session: fresh } }, actions)
    })
    expect(fresh ? newSessionInProfile : ensureGatewayProfile).toHaveBeenCalledWith('writer')
    expect(actions.openNewSessionTile).not.toHaveBeenCalled()
    expect(actions.startFreshSessionDraft).not.toHaveBeenCalled()
    expect($voiceConversationStartSessionId.get()).toBeNull()
    // Consume the profile-switch latch as the destination main composer does.
    render(mountComposer('main', false))
    expect(screen.getByTestId('main').textContent).toBe('true')
  })

  it('retains main fresh-draft behavior when workspace owns focus', async () => {
    $layoutTree.set(group(['workspace'], { id: 'wake-main' }))
    $activeTreeGroup.set('wake-main')
    const actions = { openNewSessionTile: vi.fn(), startFreshSessionDraft: vi.fn() }
    render(mountComposer('main', false))
    await act(async () => {
      handleWakeVoiceEvent({ type: 'wake.detected', payload: { start_new_session: true } }, actions)
    })
    expect(actions.startFreshSessionDraft).toHaveBeenCalledOnce()
    expect(actions.openNewSessionTile).not.toHaveBeenCalled()
    expect(screen.getByTestId('main').textContent).toBe('true')
  })

  it('starts only the third composer and keeps primary navigation intact', async () => {
    focusThirdTab()
    const actions = { openNewSessionTile: vi.fn(), startFreshSessionDraft: vi.fn() }
    render(
      <>
        {mountComposer('main', false, false, 'runtime-A')}
        {mountComposer('tile:B', false, false, 'runtime-B')}
        {mountComposer('tile:C', false, false, 'runtime-C')}
      </>
    )
    await act(async () => {
      expect(handleWakeVoiceEvent({ type: 'wake.detected', payload: { start_new_session: false } }, actions)).toBe(true)
    })
    expect($voiceConversationStartSessionId.get()).toBe('runtime-C')
    expect(screen.getByTestId('tile:C').textContent).toBe('true')
    expect(screen.getByTestId('main').textContent).toBe('false')
    expect(screen.getByTestId('tile:B').textContent).toBe('false')
    expect($selectedStoredSessionId.get()).toBe('stored-A')
    expect($activeSessionId.get()).toBe('runtime-A')
    expect(actions.startFreshSessionDraft).not.toHaveBeenCalled()
  })

  it.each([true, undefined])(
    'opens a fresh adjacent tile for flag %s, then targets its returned runtime',
    async flag => {
      focusThirdTab()
      let finish!: (id: string) => void

      const actions = {
        openNewSessionTile: vi.fn(
          () =>
            new Promise<string>(resolve => {
              finish = resolve
            })
        ),
        startFreshSessionDraft: vi.fn()
      }

      render(
        <>
          {mountComposer('main', false, false, 'runtime-A')}
          {mountComposer('tile:C', false, false, 'runtime-C')}
        </>
      )
      await act(async () => {
        handleWakeVoiceEvent({ type: 'wake.detected', payload: { start_new_session: flag } }, actions)
      })
      expect(actions.openNewSessionTile).toHaveBeenCalledWith(
        'center',
        expect.objectContaining({ anchor: 'session-tile:stored-C', listed: false })
      )
      expect(actions.startFreshSessionDraft).not.toHaveBeenCalled()
      expect(screen.getByTestId('main').textContent).toBe('false')
      expect(screen.getByTestId('tile:C').textContent).toBe('false')
      await act(async () => {
        finish('runtime-new')
      })
      expect($voiceConversationStartSessionId.get()).toBe('runtime-new')
      render(mountComposer('tile:new', false, false, 'runtime-new'))
      expect(screen.getByTestId('tile:new').textContent).toBe('true')
      expect($selectedStoredSessionId.get()).toBe('stored-A')
      expect($activeSessionId.get()).toBe('runtime-A')
    }
  )

  it('does not route a failed fresh tile create back to main', async () => {
    focusThirdTab()
    const actions = { openNewSessionTile: vi.fn(async () => undefined), startFreshSessionDraft: vi.fn() }
    render(mountComposer('main', false, false, 'runtime-A'))
    await act(async () => {
      handleWakeVoiceEvent({ type: 'wake.detected', payload: { start_new_session: true } }, actions)
    })
    expect(screen.getByTestId('main').textContent).toBe('false')
    expect(actions.startFreshSessionDraft).not.toHaveBeenCalled()
  })

  it('reserves a null target for main even if an unbound tile mounts first', async () => {
    render(
      <>
        {mountComposer('tile:unbound', false)}
        {mountComposer('main', false)}
      </>
    )
    await act(async () => {
      requestVoiceConversationStart()
    })
    expect(screen.getByTestId('tile:unbound').textContent).toBe('false')
    expect(screen.getByTestId('main').textContent).toBe('true')
  })

  it('does not consume a runtime-targeted request while its composer is disabled', async () => {
    focusThirdTab()
    const actions = { openNewSessionTile: vi.fn(), startFreshSessionDraft: vi.fn() }
    const view = render(mountComposer('tile:C', true, false, 'runtime-C'))
    await act(async () => {
      handleWakeVoiceEvent({ type: 'wake.detected', payload: { start_new_session: false } }, actions)
    })
    expect(screen.getByTestId('tile:C').textContent).toBe('false')
    view.rerender(mountComposer('tile:C', false, false, 'runtime-C'))
    expect(screen.getByTestId('tile:C').textContent).toBe('true')
  })
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
