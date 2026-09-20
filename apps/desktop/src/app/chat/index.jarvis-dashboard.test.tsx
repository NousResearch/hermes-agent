import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import type * as React from 'react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { MicRecording } from '@/app/chat/composer/hooks/use-mic-recorder'
import { $jarvisUi, publishJarvisEvent, resetJarvisSession } from '@/app/jarvis/store'
import { I18nProvider } from '@/i18n'
import { assistantTextPart, type ChatMessage } from '@/lib/chat-messages'
import { $activeGatewayProfile, $profiles } from '@/store/profile'
import {
  $activeSessionId,
  $awaitingResponse,
  $busy,
  $contextSuggestions,
  $currentCwd,
  $currentModel,
  $currentProvider,
  $freshDraftReady,
  $gatewayState,
  $messages,
  $selectedStoredSessionId,
  $sessions
} from '@/store/session'
import { setVoicePlaybackState } from '@/store/voice-playback'

const stopVoicePlayback = vi.hoisted(() => vi.fn())

const micHandle = vi.hoisted(() => ({
  cancel: vi.fn(),
  start: vi.fn(async () => undefined),
  stop: vi.fn<() => Promise<MicRecording | null>>(async () => null)
}))

vi.mock('@assistant-ui/react', async () => {
  const React = await import('react')

  const composerState = { text: '' }

  const composerRuntime = {
    getState: () => composerState,
    subscribe: () => () => undefined
  }

  const aui = {
    composer: () => ({
      setText: (value: string) => {
        composerState.text = value
      }
    })
  }

  const pass = ({ children }: { children?: React.ReactNode }) => React.createElement(React.Fragment, null, children)

  const Root = ({ children, ...props }: React.ComponentProps<'form'>) =>
    React.createElement('form', props, children)

  const Input = ({ children }: { children?: React.ReactElement }) => children ?? null

  return {
    AssistantRuntimeProvider: pass,
    ComposerPrimitive: {
      Input,
      Root,
      Unstable_TriggerPopoverRoot: pass
    },
    useAui: () => aui,
    useAuiState: <T,>(selector: (state: { composer: { text: string } }) => T) => selector({ composer: composerState }),
    useComposerRuntime: () => composerRuntime
  }
})

vi.mock('@/app/chat/tour-marker', () => ({ useTourMarker: () => undefined }))
vi.mock('@/app/hud/composer-drag', () => ({ useHudComposerDrag: () => ({ grabbing: false, onPointerDown: undefined }) }))
vi.mock('@/components/assistant-ui/thread', async () => {
  const React = await import('react')

  return {
    Thread: () => React.createElement('div', { 'data-testid': 'thread' })
  }
})
vi.mock('@/components/Backdrop', async () => {
  const React = await import('react')

  return { Backdrop: () => React.createElement('div', { 'data-testid': 'backdrop' }) }
})
vi.mock('@/components/prompt-overlays', () => ({ PromptOverlays: () => null }))
vi.mock('@/components/chat/vibe-hearts', () => ({ COMPOSER_HEART_CONFIG: {}, HeartField: () => null }))
vi.mock('@/contrib/react/slot', () => ({ Slot: () => null }))
vi.mock('@/lib/haptics', () => ({ triggerHaptic: vi.fn() }))
vi.mock('@/lib/incremental-external-store-runtime', () => ({ useIncrementalExternalStoreRuntime: () => ({}) }))
vi.mock('@/lib/model-options', () => ({
  modelOptionsQueryKey: (...parts: unknown[]) => ['model-options', ...parts],
  requestModelOptions: vi.fn(async () => ({ models: [] }))
}))
vi.mock('@/lib/thinking-sound', () => ({ startThinkingSound: vi.fn(), stopThinkingSound: vi.fn() }))
vi.mock('@/lib/voice-barge-in', () => ({ monitorSpeechDuringPlayback: vi.fn(() => vi.fn()) }))
vi.mock('@/lib/voice-playback', () => ({
  markVoicePlaybackInterrupted: vi.fn(),
  playSpeechText: vi.fn(async () => true),
  startSpeechStream: vi.fn(async () => null),
  stopVoicePlayback: () => stopVoicePlayback()
}))
vi.mock('@/store/notifications', () => ({ dismissNotification: vi.fn(), notify: vi.fn(), notifyError: vi.fn() }))
vi.mock('@/themes', () => ({
  useTheme: () => ({ availableThemes: [], themeName: 'dark' })
}))
vi.mock('./chat-drop-overlay', () => ({ ChatDropOverlay: () => null }))
vi.mock('./chat-swap-overlay', () => ({ ChatSwapOverlay: () => null, ChatSyncBadge: () => null }))
vi.mock('./composer/attachments', () => ({ AttachmentList: () => null }))
vi.mock('./composer/context-menu', () => ({ ContextMenu: () => null }))
vi.mock('./composer/controls', () => ({ ComposerControls: () => null }))
vi.mock('./composer/directive-actions', () => ({ ComposerDirectiveActions: () => null }))
vi.mock('./composer/help-hint', () => ({ HelpHint: () => null }))
vi.mock('./composer/hooks/use-at-completions', () => ({ useAtCompletions: () => ({ items: [], loading: false }) }))
vi.mock('./composer/hooks/use-composer-branch', () => ({
  useComposerBranch: () => ({
    handleBranchOff: vi.fn(),
    handleConvertBranch: vi.fn(),
    handleListBranches: vi.fn(),
    handleSwitchBranch: vi.fn(),
    openInWorktree: vi.fn()
  })
}))
vi.mock('./composer/hooks/use-composer-drop', () => ({
  useComposerDrop: () => ({
    dragActive: false,
    handleDragEnter: vi.fn(),
    handleDragLeave: vi.fn(),
    handleDragOver: vi.fn(),
    handleDrop: vi.fn(),
    handleInputDragOver: vi.fn(),
    handleInputDrop: vi.fn()
  })
}))
vi.mock('./composer/hooks/use-composer-esc-cancel', () => ({ useComposerEscCancel: vi.fn() }))
vi.mock('./composer/hooks/use-composer-metrics', () => ({
  useComposerMetrics: () => ({ compactPill: false, foldVoice: false, minimal: false, stacked: false })
}))
vi.mock('./composer/hooks/use-composer-popout', () => ({
  useComposerPopout: () => ({
    dockProximity: 0,
    dragging: false,
    handleComposerToggle: vi.fn(),
    onComposerGesturePointerDown: undefined,
    popoutAllowed: false,
    popoutPosition: { bottom: 0, right: 0 },
    poppedOut: false
  })
}))
vi.mock('./composer/hooks/use-emoji-completions', () => ({ useEmojiCompletions: () => ({ items: [], loading: false }) }))
vi.mock('./composer/hooks/use-mic-recorder', () => ({
  useMicRecorder: () => ({ handle: micHandle, level: 0, recording: false })
}))
vi.mock('./composer/hooks/use-micro-actions', () => ({ useComposerMicroActions: vi.fn() }))
vi.mock('./composer/hooks/use-slash-completions', () => ({ useSlashCompletions: () => ({ items: [], loading: false }) }))
vi.mock('./composer/hooks/use-status-presence', () => ({ useSessionStatusPresence: () => false }))
vi.mock('./composer/micro-actions', () => ({ ActionBadges: () => null, SuggestionPills: () => null }))
vi.mock('./composer/queue-panel', () => ({ QueuePanel: () => null }))
vi.mock('./composer/status-stack', () => ({ ComposerStatusStack: () => null }))
vi.mock('./composer/status-stack/coding-row', () => ({ CodingStatusRow: () => null }))
vi.mock('./composer/trigger-popover', () => ({ ComposerTriggerPopover: () => null }))
vi.mock('./composer/url-dialog', () => ({ UrlDialog: () => null }))
vi.mock('./composer/voice-activity', () => ({ VoiceActivity: () => null, VoicePlaybackActivity: () => null }))
vi.mock('./hooks/use-file-drop-zone', () => ({
  useFileDropZone: () => ({ dragKind: null, dropHandlers: {} })
}))
vi.mock('./sidebar/session-actions-menu', async () => {
  const React = await import('react')

  return {
    SessionActionsMenu: ({ children }: { children: React.ReactNode }) =>
      React.createElement('div', { 'data-testid': 'session-actions-menu' }, children)
  }
})

const { ChatView } = await import('./index')

function assistantMessage(id: string, text: string): ChatMessage {
  return {
    id,
    parts: [assistantTextPart(text)],
    role: 'assistant'
  }
}

function baseProps(overrides: Partial<React.ComponentProps<typeof ChatView>> = {}) {
  return {
    dashboard: true,
    gateway: null,
    maxVoiceRecordingSeconds: 120,
    onAddContextRef: vi.fn(),
    onAddUrl: vi.fn(),
    onAttachDroppedItems: vi.fn(),
    onAttachImageBlob: vi.fn(),
    onBranchInNewChat: vi.fn(),
    onCancel: vi.fn(),
    onDeleteSelectedSession: vi.fn(),
    onEdit: vi.fn(),
    onPasteClipboardImage: vi.fn(),
    onPickFiles: vi.fn(),
    onPickFolders: vi.fn(),
    onPickImages: vi.fn(),
    onReload: vi.fn(),
    onRemoveAttachment: vi.fn(),
    onRetryResume: vi.fn(),
    onSteer: vi.fn(),
    onSubmit: vi.fn(),
    onThreadMessagesChange: vi.fn(),
    onToggleSelectedPin: vi.fn(),
    onTranscribeAudio: vi.fn(async () => 'hello'),
    ...overrides
  }
}

function renderChatView(overrides: Partial<React.ComponentProps<typeof ChatView>> = {}) {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  const props = baseProps(overrides)

  render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter initialEntries={['/stored-1']}>
        <I18nProvider configClient={null} initialLocale="pl">
          <ChatView {...props} />
        </I18nProvider>
      </MemoryRouter>
    </QueryClientProvider>
  )

  return props
}

function setOpenSession(sessionId: string, storedSessionId = 'stored-1') {
  $activeSessionId.set(sessionId)
  $selectedStoredSessionId.set(storedSessionId)
  $sessions.set([{ id: storedSessionId, message_count: 1, title: 'Stable chat' } as never])
}

function publishTaskPhase(type: string, at: number, sessionId = $activeSessionId.get() ?? 'runtime-1') {
  act(() =>
    publishJarvisEvent({
      at,
      sessionId,
      taskId: 'task-1',
      type
    })
  )
}

async function startDashboardVoice() {
  const controls = within(screen.getByTestId('jarvis-voice-controls'))

  await act(async () => {
    fireEvent.click(controls.getByRole('button', { name: 'Zacznij słuchać' }))
  })

  await waitFor(() => expect(micHandle.start).toHaveBeenCalled())
  expect(await controls.findByRole('button', { name: 'Przestań słuchać' })).toBeTruthy()
}

function mockViewport() {
  Object.defineProperty(window, 'matchMedia', {
    configurable: true,
    value: vi.fn().mockImplementation((query: string) => ({
      matches: query === '(max-width: 767px)' || query === '(max-width: 1149px)' ? false : false,
      media: query,
      onchange: null,
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      addListener: vi.fn(),
      removeListener: vi.fn(),
      dispatchEvent: vi.fn()
    }))
  })
}

describe('ChatView Jarvis dashboard seam', () => {
  beforeEach(() => {
    mockViewport()
    $gatewayState.set('open')
    $awaitingResponse.set(false)
    $busy.set(false)
    $contextSuggestions.set([])
    $currentCwd.set('/work')
    $currentModel.set('test-model')
    $currentProvider.set('test-provider')
    $freshDraftReady.set(false)
    $messages.set([assistantMessage('assistant-1', 'Stable historical answer')])
    $activeGatewayProfile.set('default')
    $profiles.set([{ display_name: 'Ada', name: 'default' } as never])
    setOpenSession('runtime-1')
    resetJarvisSession('runtime-1')
    setVoicePlaybackState({ audioElement: null, messageId: null, sequence: 0, source: null, status: 'idle' })
    vi.clearAllMocks()
    micHandle.start.mockResolvedValue(undefined)
    micHandle.stop.mockResolvedValue(null)
  })

  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
    $activeSessionId.set(null)
    $awaitingResponse.set(false)
    $busy.set(false)
    $contextSuggestions.set([])
    $currentCwd.set('')
    $currentModel.set('')
    $currentProvider.set('')
    $freshDraftReady.set(false)
    $gatewayState.set('idle')
    $messages.set([])
    $activeGatewayProfile.set('default')
    $profiles.set([])
    $selectedStoredSessionId.set(null)
    $sessions.set([])
    resetJarvisSession(null)
    setVoicePlaybackState({ audioElement: null, messageId: null, sequence: 0, source: null, status: 'idle' })
  })

  it('uses the real main composer voice hook to render and stop listening', async () => {
    renderChatView()

    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: 'Zacznij słuchać' }))
    })

    await waitFor(() => expect(micHandle.start).toHaveBeenCalledTimes(1))
    expect(await screen.findByRole('button', { name: 'Przestań słuchać' })).toBeTruthy()
    expect(screen.getByTestId('jarvis-core').getAttribute('data-voice')).toBe('listening')

    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: 'Przestań słuchać' }))
    })

    await waitFor(() => expect(micHandle.cancel).toHaveBeenCalledTimes(1))
    expect(await screen.findByRole('button', { name: 'Zacznij słuchać' })).toBeTruthy()
  })

  it('routes stop speaking to voice playback and stop task to ChatView onCancel', async () => {
    const props = renderChatView()

    act(() =>
      setVoicePlaybackState({
        audioElement: null,
        messageId: 'm1',
        sequence: 1,
        source: 'voice-conversation',
        status: 'speaking'
      })
    )

    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: 'Przestań mówić' }))
    })

    expect(stopVoicePlayback).toHaveBeenCalledTimes(1)
    expect(props.onCancel).not.toHaveBeenCalled()

    publishTaskPhase('task.running', 1)
    await waitFor(() => expect(screen.getByRole<HTMLButtonElement>('button', { name: 'Zatrzymaj zadanie' }).disabled).toBe(false))

    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: 'Zatrzymaj zadanie' }))
    })

    expect(props.onCancel).toHaveBeenCalledTimes(1)
  })

  it('routes stop speaking to voice playback while audio is preparing', async () => {
    const props = renderChatView()

    act(() =>
      setVoicePlaybackState({
        audioElement: null,
        messageId: 'm1',
        sequence: 1,
        source: 'voice-conversation',
        status: 'preparing'
      })
    )

    await waitFor(() => expect(screen.getByRole<HTMLButtonElement>('button', { name: 'Przestań mówić' }).disabled).toBe(false))

    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: 'Przestań mówić' }))
    })

    expect(stopVoicePlayback).toHaveBeenCalledTimes(1)
    expect(props.onCancel).not.toHaveBeenCalled()
  })

  it('enables cancel only for active Jarvis task phases in the mounted dashboard', () => {
    renderChatView()
    const button = () => screen.getByRole<HTMLButtonElement>('button', { name: 'Zatrzymaj zadanie' })

    expect(button().disabled).toBe(true)

    publishTaskPhase('task.planning', 1)
    expect(button().disabled).toBe(false)

    publishTaskPhase('task.running', 2)
    expect(button().disabled).toBe(false)

    publishTaskPhase('task.approval', 3)
    expect(button().disabled).toBe(false)

    publishTaskPhase('task.cancelling', 4)
    expect(button().disabled).toBe(true)

    publishTaskPhase('task.cancelled', 5)
    expect(button().disabled).toBe(true)

    publishTaskPhase('task.failed', 6)
    expect(button().disabled).toBe(true)

    publishTaskPhase('task.verified', 7)
    expect(button().disabled).toBe(true)
  })

  it('resets stale Jarvis UI when the foreground session changes', async () => {
    renderChatView()

    publishTaskPhase('task.running', 1)
    expect(screen.getByRole<HTMLButtonElement>('button', { name: 'Zatrzymaj zadanie' }).disabled).toBe(false)

    act(() => setOpenSession('runtime-2', 'stored-2'))

    await waitFor(() => expect($jarvisUi.get()).toMatchObject({ sessionId: 'runtime-2', task: { phase: 'idle' } }))
    expect(screen.getByRole<HTMLButtonElement>('button', { name: 'Zatrzymaj zadanie' }).disabled).toBe(true)
  })

  it('stops the active main voice conversation exactly once on a foreground session switch', async () => {
    renderChatView()
    const controls = () => within(screen.getByTestId('jarvis-voice-controls'))

    await startDashboardVoice()
    expect(micHandle.cancel).not.toHaveBeenCalled()

    act(() => {
      publishJarvisEvent({
        at: 1,
        sessionId: 'background-session',
        taskId: 'background-task',
        type: 'task.running'
      })
      $freshDraftReady.set(true)
    })

    expect(micHandle.cancel).not.toHaveBeenCalled()
    expect(controls().getByRole('button', { name: 'Przestań słuchać' })).toBeTruthy()

    act(() => setOpenSession('runtime-2', 'stored-2'))

    await waitFor(() => expect(micHandle.cancel).toHaveBeenCalledTimes(1))
    expect(await controls().findByRole('button', { name: 'Zacznij słuchać' })).toBeTruthy()

    act(() => setOpenSession('runtime-2', 'stored-2'))
    expect(micHandle.cancel).toHaveBeenCalledTimes(1)
  })

  it('stops the active main voice conversation exactly once on a foreground profile switch', async () => {
    renderChatView()
    const controls = () => within(screen.getByTestId('jarvis-voice-controls'))

    await startDashboardVoice()

    act(() => {
      $profiles.set([
        { display_name: 'Ada', name: 'default' },
        { display_name: 'Research', name: 'research' }
      ] as never)
      $activeGatewayProfile.set('research')
    })

    await waitFor(() => expect(micHandle.cancel).toHaveBeenCalledTimes(1))
    expect(await controls().findByRole('button', { name: 'Zacznij słuchać' })).toBeTruthy()

    act(() => $activeGatewayProfile.set('research'))
    expect(micHandle.cancel).toHaveBeenCalledTimes(1)
  })

  it('resets stale Jarvis UI when the active profile changes', async () => {
    renderChatView()

    publishTaskPhase('task.running', 1)
    expect(screen.getByRole<HTMLButtonElement>('button', { name: 'Zatrzymaj zadanie' }).disabled).toBe(false)

    act(() => {
      $profiles.set([
        { display_name: 'Ada', name: 'default' },
        { display_name: 'Research', name: 'research' }
      ] as never)
      $activeGatewayProfile.set('research')
    })

    await waitFor(() => expect($jarvisUi.get()).toMatchObject({ sessionId: 'runtime-1', task: { phase: 'idle' } }))
    expect(screen.getByRole<HTMLButtonElement>('button', { name: 'Zatrzymaj zadanie' }).disabled).toBe(true)
  })

  it('ignores background Jarvis events after the foreground session is scoped', () => {
    renderChatView()

    publishTaskPhase('task.running', 1, 'background-session')

    expect($jarvisUi.get()).toMatchObject({ sessionId: 'runtime-1', task: { phase: 'idle' } })
    expect(screen.getByRole<HTMLButtonElement>('button', { name: 'Zatrzymaj zadanie' }).disabled).toBe(true)
  })
})
