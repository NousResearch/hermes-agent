import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { useRef } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { ReviewShipBar } from '@/app/right-sidebar/review/ship-bar'
import { PaneVisibleContext } from '@/components/pane-shell/pane-visibility'
import type { HermesReviewFile } from '@/global'
import { I18nProvider } from '@/i18n'
import { en } from '@/i18n/en'
import { $parkedQueueSessions, $queuedPromptsBySession, getQueuedPrompts } from '@/store/composer-queue'
import { continueReplyPause, endContinuedReplyPause, endReplyPause, markReplyPaused } from '@/store/paused-replies'
import { $reviewFiles, $reviewScope, $reviewScopeTarget, $reviewShipInfo } from '@/store/review'
import { setSessionsLoading } from '@/store/session'
import { $shellChoice } from '@/store/shell-variant'

import type { QueueEditState } from './composer-utils'
import { useComposerQueue } from './hooks/use-composer-queue'
import { useComposerSubmit } from './hooks/use-composer-submit'
import { ComposerScopeProvider, ComposerSurfaceProvider, MAIN_COMPOSER_SCOPE } from './scope'
import type { ChatBarProps } from './types'

// A reply paused for its automatic continuation (store/paused-replies.ts) holds what the composer sends
// until the continuation has completed. A request another surface sends through the composer is held the
// same way. Here the review pane's real button asks, and the composer's two real hooks (the submit engine
// that takes the request, the queue that holds and drains it) answer, wired as in composer/index.tsx.

const STORED = 'stored-session-paused-requests'
const RUNTIME = 'rt-session-paused-requests'
const SURFACE = 'surface-paused-requests'
const coding = en.statusStack.coding

interface Sent {
  onSteer: ReturnType<typeof vi.fn<NonNullable<ChatBarProps['onSteer']>>>
  onSteerHidden: ReturnType<typeof vi.fn<NonNullable<ChatBarProps['onSteerHidden']>>>
  onSubmit: ReturnType<typeof vi.fn<ChatBarProps['onSubmit']>>
}

function Composer({ busy, sent }: { busy: boolean; sent: Sent }) {
  const draftRef = useRef('unsent draft')
  const draftScopeRef = useRef<string | null>(STORED)
  const editorRef = useRef<HTMLDivElement | null>(null)
  const queueEditRef = useRef<QueueEditState | null>(null)
  const activeQueueSessionKeyRef = useRef<string | null>(STORED)
  const ui = useRef({ clearDraft: vi.fn(), focusInput: vi.fn(), loadIntoComposer: vi.fn(), onCancel: vi.fn() }).current
  const draft = useRef({ setComposerText: vi.fn(), stashAt: vi.fn() }).current

  const queue = useComposerQueue({
    activeQueueSessionKey: STORED,
    attachments: [],
    busy,
    clearDraft: ui.clearDraft,
    draftRef,
    focusInput: ui.focusInput,
    loadIntoComposer: ui.loadIntoComposer,
    onCancel: ui.onCancel,
    onSteer: sent.onSteer,
    onSubmit: sent.onSubmit,
    queueEditRef,
    queueSessionKey: STORED,
    sessionId: RUNTIME
  })

  useComposerSubmit({
    activeQueueSessionKey: STORED,
    activeQueueSessionKeyRef,
    attachments: [],
    busy,
    clearDraft: ui.clearDraft,
    disabled: false,
    draftRef,
    draftScopeRef,
    drainNextQueued: queue.drainNextQueued,
    editorRef,
    exitQueuedEdit: queue.exitQueuedEdit,
    focusInput: ui.focusInput,
    inputDisabled: false,
    loadIntoComposer: ui.loadIntoComposer,
    onCancel: ui.onCancel,
    onSteer: sent.onSteer,
    onSteerHidden: sent.onSteerHidden,
    onSubmit: sent.onSubmit,
    queueCurrentDraft: queue.queueCurrentDraft,
    queueEdit: queue.queueEdit,
    queuedPrompts: queue.queuedPrompts,
    sessionId: RUNTIME,
    setComposerText: draft.setComposerText,
    stashAt: draft.stashAt
  })

  return <div data-draft={draftRef.current} data-testid="composer" />
}

function Screen({ busy, sent }: { busy: boolean; sent: Sent }) {
  return (
    <I18nProvider configClient={null} initialLocale="en">
      <ComposerScopeProvider value={MAIN_COMPOSER_SCOPE}>
        <ComposerSurfaceProvider value={SURFACE}>
          <PaneVisibleContext.Provider value>
            <div data-composer-surface-id={SURFACE} data-composer-target="main">
              <Composer busy={busy} sent={sent} />
            </div>
          </PaneVisibleContext.Provider>
        </ComposerSurfaceProvider>
      </ComposerScopeProvider>
      <ReviewShipBar />
    </I18nProvider>
  )
}

const file: HermesReviewFile = { added: 1, path: 'a.ts', removed: 0, staged: false, status: 'M' }
const queuedTexts = () => getQueuedPrompts(STORED).map(({ text }) => text)
const settle = () => act(async () => void (await new Promise(resolve => setTimeout(resolve, 25))))

function renderScreen(busy: boolean) {
  const sent: Sent = {
    onSteer: vi.fn<NonNullable<ChatBarProps['onSteer']>>(async () => true),
    onSteerHidden: vi.fn<NonNullable<ChatBarProps['onSteerHidden']>>(async () => true),
    onSubmit: vi.fn<ChatBarProps['onSubmit']>(async () => true)
  }

  const view = render(<Screen busy={busy} sent={sent} />)

  return { sent, setBusy: (next: boolean) => view.rerender(<Screen busy={next} sent={sent} />) }
}

const askHermesToShip = () => fireEvent.click(screen.getByText(coding.agentShip))

describe("the review pane's ship request while the chat's reply is paused for its continuation", () => {
  beforeEach(() => {
    window.localStorage.clear()
    $queuedPromptsBySession.set({})
    $parkedQueueSessions.set({})
    $reviewFiles.set([file])
    $reviewShipInfo.set({ ghReady: false, pr: null })
    $reviewScope.set('uncommitted')
    $reviewScopeTarget.set('main')
    setSessionsLoading(false)
    // Classic's wording; the redesign names the agent instead of Hermes.
    $shellChoice.set(false)
  })

  afterEach(() => {
    cleanup()
    endReplyPause(STORED)
    $queuedPromptsBySession.set({})
    $parkedQueueSessions.set({})
    $reviewFiles.set([])
    setSessionsLoading(true)
    $shellChoice.set(null)
    vi.restoreAllMocks()
  })

  it('is sent at once when no reply is paused', async () => {
    const { sent } = renderScreen(false)

    askHermesToShip()

    await waitFor(() => expect(sent.onSubmit).toHaveBeenCalledTimes(1))
    expect(sent.onSubmit.mock.calls[0]![0]).toBe(coding.agentShipPrompt)
    expect(queuedTexts()).toEqual([])
  })

  it('waits in the queue while the reply waits for its continuation and through it, then goes once', async () => {
    markReplyPaused(STORED)
    const { sent, setBusy } = renderScreen(false)

    askHermesToShip()
    await settle()

    // Not ahead of the continuation.
    expect(queuedTexts()).toEqual([coding.agentShipPrompt])
    expect(sent.onSubmit).not.toHaveBeenCalled()

    // The continuation starts and runs as a turn of its own: not into it either.
    act(() => continueReplyPause(STORED))
    setBusy(true)
    await settle()

    expect(queuedTexts()).toEqual([coding.agentShipPrompt])
    expect(sent.onSubmit).not.toHaveBeenCalled()
    expect(sent.onSteer).not.toHaveBeenCalled()

    // The continuation completes: the held request is the next turn, once.
    act(() => endContinuedReplyPause(STORED))
    setBusy(false)

    await waitFor(() => expect(sent.onSubmit).toHaveBeenCalledTimes(1))
    expect(sent.onSubmit.mock.calls[0]![0]).toBe(coding.agentShipPrompt)
    await waitFor(() => expect(queuedTexts()).toEqual([]))
    await settle()

    expect(sent.onSubmit).toHaveBeenCalledTimes(1)
    expect(sent.onSteer).not.toHaveBeenCalled()
    expect(sent.onSteerHidden).not.toHaveBeenCalled()
    expect(screen.getByTestId('composer').dataset.draft).toBe('unsent draft')
  })

  it('is not steered into a continuation that is already running, and goes once after it', async () => {
    markReplyPaused(STORED)
    continueReplyPause(STORED)
    const { sent, setBusy } = renderScreen(true)

    askHermesToShip()
    await settle()

    expect(sent.onSteer).not.toHaveBeenCalled()
    expect(sent.onSubmit).not.toHaveBeenCalled()
    expect(queuedTexts()).toEqual([coding.agentShipPrompt])

    act(() => endContinuedReplyPause(STORED))
    setBusy(false)

    await waitFor(() => expect(sent.onSubmit).toHaveBeenCalledTimes(1))
    expect(sent.onSubmit.mock.calls[0]![0]).toBe(coding.agentShipPrompt)
    await waitFor(() => expect(queuedTexts()).toEqual([]))
    await settle()

    expect(sent.onSubmit).toHaveBeenCalledTimes(1)
    expect(sent.onSteer).not.toHaveBeenCalled()
  })
})
