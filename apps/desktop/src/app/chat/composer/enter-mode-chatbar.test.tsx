import { cleanup, fireEvent, render, waitFor } from '@testing-library/react'
import type { PropsWithChildren } from 'react'
import type * as ReactModule from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'
import { mainComposerScope } from '@/store/composer'
import { $composerEnterSends } from '@/store/composer-prefs'

import type { QueueEditState } from './composer-utils'
import type { useComposerBranch } from './hooks/use-composer-branch'
import type { useComposerDraft } from './hooks/use-composer-draft'
import type { useComposerDrop } from './hooks/use-composer-drop'
import type { useComposerMetrics } from './hooks/use-composer-metrics'
import type { useComposerPopout } from './hooks/use-composer-popout'
import type { useComposerQueue } from './hooks/use-composer-queue'
import type { useComposerTrigger } from './hooks/use-composer-trigger'
import type { useComposerUrlDialog } from './hooks/use-composer-url-dialog'
import type { useComposerVoice } from './hooks/use-composer-voice'

import { ChatBar } from './index'

// Keep the unavoidable ChatBar isolation stubs structurally tied to the real
// hooks. Any required return-field change now fails typecheck instead of letting
// this integration test silently drift from the production component surface.
type ComposerBranchResult = ReturnType<typeof useComposerBranch>
type ComposerDraftResult = ReturnType<typeof useComposerDraft>
type ComposerDropResult = ReturnType<typeof useComposerDrop>
type ComposerMetricsResult = ReturnType<typeof useComposerMetrics>
type ComposerPopoutResult = ReturnType<typeof useComposerPopout>
type ComposerQueueResult = ReturnType<typeof useComposerQueue>
type ComposerTriggerResult = ReturnType<typeof useComposerTrigger>
type ComposerUrlDialogResult = ReturnType<typeof useComposerUrlDialog>
type ComposerVoiceResult = ReturnType<typeof useComposerVoice>

const queueCurrentDraftMock = vi.hoisted(() => vi.fn(() => true))
const queueStateMock = vi.hoisted(() => ({ queueEdit: null as QueueEditState | null }))

vi.mock('@assistant-ui/react', () => ({
  ComposerPrimitive: {
    Input: ({ children }: PropsWithChildren<{ asChild?: boolean }>) => children,
    Root: ({ children, ...props }: PropsWithChildren<Record<string, unknown>>) => <form {...props}>{children}</form>,
    Unstable_TriggerPopoverRoot: ({ children }: PropsWithChildren) => <>{children}</>
  }
}))

vi.mock('@/lib/haptics', () => ({ triggerHaptic: vi.fn() }))
vi.mock('@/themes', () => ({ useTheme: () => ({ availableThemes: [], themeName: 'default' }) }))

vi.mock('./attachments', () => ({ AttachmentList: () => null }))
vi.mock('./context-menu', () => ({ ContextMenu: () => null }))
vi.mock('./controls', () => ({ ComposerControls: () => null }))
vi.mock('./status-stack', () => ({ ComposerStatusStack: ({ children }: PropsWithChildren) => <>{children}</> }))
vi.mock('./status-stack/coding-row', () => ({ CodingStatusRow: () => null }))
vi.mock('./trigger-popover', () => ({ ComposerTriggerPopover: () => null }))
vi.mock('./url-dialog', () => ({ UrlDialog: () => null }))
vi.mock('./voice-activity', () => ({ VoiceActivity: () => null, VoicePlaybackActivity: () => null }))
vi.mock('./queue-panel', () => ({ QueuePanel: () => null }))

vi.mock('./hooks/use-at-completions', () => ({ useAtCompletions: () => ({ adapter: null, loading: false }) }))
vi.mock('./hooks/use-slash-completions', () => ({ useSlashCompletions: () => ({ adapter: null, loading: false }) }))
vi.mock('./hooks/use-status-presence', () => ({ useSessionStatusPresence: () => false }))
vi.mock('./hooks/use-composer-esc-cancel', () => ({ useComposerEscCancel: () => undefined }))
vi.mock('./hooks/use-composer-placeholder', () => ({ useComposerPlaceholder: () => 'Message Hermes' }))
vi.mock('./hooks/use-composer-metrics', () => ({
  useComposerMetrics: () =>
    ({ compactPill: false, foldVoice: false, minimal: false, stacked: false, tight: false }) satisfies ComposerMetricsResult
}))
vi.mock('./hooks/use-composer-popout', () => ({
  useComposerPopout: () =>
    ({
      dockProximity: 0,
      dragging: false,
      handleComposerToggle: vi.fn(),
      onComposerGesturePointerDown: vi.fn(),
      popoutAllowed: false,
      popoutPosition: { bottom: 0, right: 0 },
      poppedOut: false
    }) satisfies ComposerPopoutResult
}))
vi.mock('./hooks/use-composer-drop', () => ({
  useComposerDrop: () =>
    ({
      dragActive: false,
      handleDragEnter: vi.fn(),
      handleDragLeave: vi.fn(),
      handleDragOver: vi.fn(),
      handleDrop: vi.fn(),
      handleInputDragOver: vi.fn(),
      handleInputDrop: vi.fn()
    }) satisfies ComposerDropResult
}))
vi.mock('./hooks/use-composer-branch', () => ({
  useComposerBranch: () =>
    ({
      handleBranchOff: vi.fn(),
      handleConvertBranch: vi.fn(),
      handleListBranches: vi.fn(),
      handleSwitchBranch: vi.fn(),
      openInWorktree: vi.fn()
    }) satisfies ComposerBranchResult
}))
vi.mock('./hooks/use-composer-url-dialog', async () => {
  const React = (await vi.importActual('react')) as typeof ReactModule

  return {
    useComposerUrlDialog: () =>
      ({
        openUrlDialog: vi.fn(),
        setUrlOpen: vi.fn(),
        setUrlValue: vi.fn(),
        submitUrl: vi.fn(),
        urlInputRef: React.createRef<HTMLInputElement>(),
        urlOpen: false,
        urlValue: ''
      }) satisfies ComposerUrlDialogResult
  }
})
vi.mock('./hooks/use-composer-voice', () => ({
  useComposerVoice: () =>
    ({
      conversation: {
        end: vi.fn(),
        level: 0,
        muted: false,
        start: vi.fn(),
        status: 'idle',
        stopTurn: vi.fn(),
        toggleMute: vi.fn()
      },
      dictate: vi.fn(),
      endConversation: vi.fn(),
      handleToggleAutoSpeak: vi.fn(),
      startConversation: vi.fn(),
      voiceActivityState: { elapsedSeconds: 0, level: 0, status: 'idle' },
      voiceConversationActive: false,
      voiceStatus: 'idle'
    }) satisfies ComposerVoiceResult
}))
vi.mock('./hooks/use-composer-trigger', async importOriginal => {
  const React = (await vi.importActual('react')) as typeof ReactModule
  const actual = await importOriginal<Record<string, unknown>>()

  return {
    ...actual,
    useComposerTrigger: () =>
      ({
        argStageEmpty: false,
        ascendTriggerPath: vi.fn(() => false),
        closeTrigger: vi.fn(),
        commitTypedSlashDirective: vi.fn(),
        moveTriggerActive: vi.fn(),
        refreshTrigger: vi.fn(),
        replaceTriggerWithChip: vi.fn(),
        setTriggerActive: vi.fn(),
        slashFreeTextArgStage: false,
        trigger: null,
        triggerActive: 0,
        triggerActiveExplicit: false,
        triggerItems: [],
        triggerKeyConsumedRef: React.useRef(false),
        triggerLoading: false
      }) satisfies ComposerTriggerResult
  }
})
vi.mock('./hooks/use-composer-queue', () => ({
  useComposerQueue: () =>
    ({
      beginQueuedEdit: vi.fn(),
      drainNextQueued: vi.fn(),
      editingQueuedPrompt: null,
      exitQueuedEdit: vi.fn(() => false),
      queueCurrentDraft: queueCurrentDraftMock,
      queueEdit: queueStateMock.queueEdit,
      queueParked: false,
      queuedPrompts: [],
      sendQueuedNow: vi.fn(),
      steerQueuedNow: vi.fn(),
      stepQueuedEdit: vi.fn(() => false)
    }) satisfies ComposerQueueResult
}))
vi.mock('./hooks/use-composer-draft', async () => {
  const React = (await vi.importActual('react')) as typeof ReactModule

  return {
    useComposerDraft: () => {
      const activeQueueSessionKeyRef = React.useRef<string | null>(null)
      const draftRef = React.useRef('')
      const editorRef = React.useRef<HTMLDivElement | null>(null)
      const sessionIdRef = React.useRef<string | null>(null)
      const [, forceRender] = React.useState(0)

      const setComposerText = (value: string) => {
        draftRef.current = value
        forceRender(v => v + 1)
      }

      const clearDraft = () => {
        draftRef.current = ''
        editorRef.current?.replaceChildren()
        forceRender(v => v + 1)
      }

      return {
        activeQueueSessionKeyRef,
        clearDraft,
        draftRef,
        editorRef,
        focusInput: vi.fn(),
        hasText: draftRef.current.trim().length > 0,
        insertInlineRefs: vi.fn(() => false),
        insertText: vi.fn(),
        isHelpHint: false,
        isSteerableText: draftRef.current.trim().length > 0,
        loadIntoComposer: vi.fn(),
        requestMainFocus: vi.fn(),
        sessionIdRef,
        setComposerText,
        stashAt: vi.fn(),
        syncDraftFromEditor: () => draftRef.current
      } satisfies ComposerDraftResult
    }
  }
})

const state = {
  model: { canSwitch: false, model: 'test-model', provider: 'test-provider' },
  tools: { enabled: false, label: 'Tools' },
  voice: { active: false, enabled: false }
}

function renderChatBar(props: Partial<Parameters<typeof ChatBar>[0]> = {}) {
  const onSubmit = vi.fn(() => true)
  const onSteer = vi.fn(() => true)

  const rendered = render(
    <I18nProvider configClient={null} initialLocale="en">
      <ChatBar
        busy={false}
        disabled={false}
        onCancel={vi.fn()}
        onSteer={onSteer}
        onSubmit={onSubmit}
        sessionId="session-1"
        state={state}
        {...props}
      />
    </I18nProvider>
  )

  return { ...rendered, editor: rendered.getByRole('textbox'), onSteer, onSubmit }
}

afterEach(() => {
  cleanup()
  queueCurrentDraftMock.mockClear()
  queueStateMock.queueEdit = null
  $composerEnterSends.set(true)
  mainComposerScope.clear()
})

describe('ChatBar composer Enter mode', () => {
  it('uses the post-refactor ChatBar keydown path to send with Enter by default', async () => {
    const { editor, onSubmit } = renderChatBar()

    editor.textContent = 'send me'
    fireEvent.keyDown(editor, { key: 'Enter' })

    await waitFor(() =>
      expect(onSubmit).toHaveBeenCalledWith('send me', { attachments: [], composerScope: null })
    )
  })

  it('also submits with Cmd/Ctrl+Enter while idle in the default mode', async () => {
    const { editor, onSubmit } = renderChatBar()

    editor.textContent = 'send with chord'
    fireEvent.keyDown(editor, { key: 'Enter', metaKey: true })

    await waitFor(() =>
      expect(onSubmit).toHaveBeenCalledWith('send with chord', { attachments: [], composerScope: null })
    )
  })

  it('uses config-driven multiline mode: Enter inserts a newline and does not submit', () => {
    $composerEnterSends.set(false)
    const { editor, onSubmit } = renderChatBar()

    editor.textContent = 'line one'
    fireEvent.keyDown(editor, { key: 'Enter' })

    expect(onSubmit).not.toHaveBeenCalled()
    expect(editor.innerHTML).toContain('<br>')
  })

  it('uses config-driven multiline mode: Cmd/Ctrl+Enter submits live DOM text', async () => {
    $composerEnterSends.set(false)
    const { editor, onSubmit } = renderChatBar()

    editor.textContent = 'send from multiline mode'
    fireEvent.keyDown(editor, { key: 'Enter', metaKey: true })

    await waitFor(() =>
      expect(onSubmit).toHaveBeenCalledWith('send from multiline mode', {
        attachments: [],
        composerScope: null
      })
    )
  })

  it('uses config-driven multiline mode: Cmd/Ctrl+Enter queues a normal prompt while busy', () => {
    $composerEnterSends.set(false)
    const { editor, onSteer, onSubmit } = renderChatBar({ busy: true })

    editor.textContent = 'follow up later'
    fireEvent.keyDown(editor, { key: 'Enter', metaKey: true })

    expect(queueCurrentDraftMock).toHaveBeenCalledOnce()
    expect(onSteer).not.toHaveBeenCalled()
    expect(onSubmit).not.toHaveBeenCalled()
  })

  it('uses config-driven multiline mode: Cmd/Ctrl+Enter executes slash commands immediately while busy', async () => {
    $composerEnterSends.set(false)
    const { editor, onSubmit } = renderChatBar({ busy: true })

    editor.textContent = '/help'
    fireEvent.keyDown(editor, { key: 'Enter', metaKey: true })

    await waitFor(() => expect(onSubmit).toHaveBeenCalledWith('/help', { composerScope: null }))
    expect(queueCurrentDraftMock).not.toHaveBeenCalled()
  })

  it('does not send or insert a newline when Enter confirms an IME composition', () => {
    $composerEnterSends.set(false)
    const { editor, onSubmit } = renderChatBar()

    fireEvent.compositionStart(editor)
    editor.textContent = '日本語'
    fireEvent.keyDown(editor, { key: 'Enter' })

    expect(onSubmit).not.toHaveBeenCalled()
    expect(editor.textContent).toBe('日本語')
  })

  it('uses config-driven multiline mode: Shift+Enter steers a live run when steerable', () => {
    $composerEnterSends.set(false)
    const { editor, onSteer, onSubmit } = renderChatBar({ busy: true })

    editor.textContent = 'nudge'
    fireEvent.keyDown(editor, { key: 'Enter', shiftKey: true })

    expect(onSteer).toHaveBeenCalledWith('nudge')
    expect(onSubmit).not.toHaveBeenCalled()
  })

  it('keeps Shift+Enter as a newline while editing a queued prompt', () => {
    $composerEnterSends.set(false)
    queueStateMock.queueEdit = {
      attachments: [],
      draft: 'original queued prompt',
      entryId: 'queued-1',
      sessionKey: 'session-1'
    }
    const { editor, onSteer, onSubmit } = renderChatBar({ busy: true })

    editor.textContent = 'edit queued prompt'
    fireEvent.keyDown(editor, { key: 'Enter', shiftKey: true })

    expect(onSteer).not.toHaveBeenCalled()
    expect(onSubmit).not.toHaveBeenCalled()
    expect(editor.innerHTML).toContain('<br>')
  })
})
