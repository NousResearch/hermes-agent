import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import type { ThreadMessage } from '@assistant-ui/react'
import { atom } from 'nanostores'
import { Profiler, type ProfilerOnRenderCallback, useState } from 'react'
import { MemoryRouter, useNavigate } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { commitReceiptContentSignature, type ThreadCommitReceipt } from '@/components/assistant-ui/thread/list'
import { PaneVisibleContext } from '@/components/pane-shell/pane-visibility'
import { assistantTextPart, type ChatMessage } from '@/lib/chat-messages'
import { createClientSessionState } from '@/lib/chat-runtime'
import { $sessionStates } from '@/store/session-states'
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

import type { RevealExpectation } from './index'
import { type SessionView, SessionViewProvider } from './session-view'

const threadRenderCount = vi.hoisted(() => ({ current: 0 }))
const revealTrace = vi.hoisted(() => ({ counter: 0, sequence: [] as TraceEvent[] }))
// Lets a test hold back Thread receipts so a stale pre-switch receipt can be
// proven incapable of revealing a switched-to session.
const receiptGate = vi.hoisted(() => ({
  fallbackPublicationIdentity: 'primary:runtime-1',
  suppressed: false,
  suppressedPublicationIdentity: null as string | null
}))

type TraceEvent =
  | { at: number; hidden: boolean; kind: 'commit'; messageCount: number; revision: number }
  | {
      at: number
      chainSignature: string
      headMessageId: string | null
      kind: 'receipt'
      publicationIdentity: string
      revision: number
    }

vi.mock('@/components/assistant-ui/thread', async () => {
  const React = await import('react')
  const { useLayoutEffect } = await import('react')
  const { useThread } = await import('@assistant-ui/react')
  const listModule = (await import('@/components/assistant-ui/thread/list')) as Record<string, unknown>
  const usePublicationIdentity =
    (listModule.useThreadPublicationIdentity as (() => string) | undefined) ??
    (() => receiptGate.fallbackPublicationIdentity)

  return {
    Thread: (props: {
      onCommitReceipt?: (receipt: ThreadCommitReceipt) => void
      resumePublicationRevision?: number
    }) => {
      threadRenderCount.current += 1
      const { messages } = useThread()
      const publicationIdentity = usePublicationIdentity()
      const committedMessages = messages.filter(message => message.metadata?.isOptimistic !== true)

      useLayoutEffect(() => {
        if (
          !props.onCommitReceipt ||
          receiptGate.suppressed ||
          receiptGate.suppressedPublicationIdentity === publicationIdentity
        ) {
          return
        }

        const receipt = {
          revision: props.resumePublicationRevision ?? 0,
          chainSignature: committedMessages.map(message => message.id).join('\n'),
          headMessage: committedMessages.at(-1) ?? null,
          contentSignature: commitReceiptContentSignature(committedMessages),
          publicationIdentity,
          complete: true
        } as ThreadCommitReceipt & { publicationIdentity: string }

        revealTrace.sequence.push({
          at: revealTrace.counter++,
          kind: 'receipt',
          revision: receipt.revision,
          chainSignature: receipt.chainSignature,
          headMessageId: receipt.headMessage?.id ?? null,
          publicationIdentity
        })
        props.onCommitReceipt(receipt)
      }, [committedMessages, props, publicationIdentity])

      return React.createElement('div', {
        'data-testid': 'thread',
        'data-message-count': String(messages.length),
        'data-revision': String(props.resumePublicationRevision ?? 0)
      })
    }
  }
})

vi.mock('@/components/Backdrop', async () => {
  const React = await import('react')

  return { Backdrop: () => React.createElement('div', { 'data-testid': 'backdrop' }) }
})

vi.mock('@/components/prompt-overlays', () => ({ PromptOverlays: () => null }))
vi.mock('@/components/chat/vibe-hearts', () => ({ COMPOSER_HEART_CONFIG: {}, HeartField: () => null }))
vi.mock('@/lib/model-options', () => ({
  modelOptionsQueryKey: (...parts: unknown[]) => ['model-options', ...parts],
  requestModelOptions: vi.fn(async () => ({ models: [] }))
}))
vi.mock('./chat-drop-overlay', () => ({ ChatDropOverlay: () => null }))
vi.mock('./chat-swap-overlay', () => ({ ChatSwapOverlay: () => null }))
vi.mock('./composer', () => ({ ChatBar: () => null, ChatBarFallback: () => null }))
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

const { ChatView, revealMatchesExpectation } = await import('./index')

function assistantMessage(id: string, text: string): ChatMessage {
  return {
    id,
    parts: [assistantTextPart(text)],
    role: 'assistant'
  }
}

describe('ChatView render isolation', () => {
  beforeEach(() => {
    threadRenderCount.current = 0
    revealTrace.counter = 0
    revealTrace.sequence.length = 0
    receiptGate.fallbackPublicationIdentity = 'primary:runtime-1'
    receiptGate.suppressed = false
    receiptGate.suppressedPublicationIdentity = null
    $activeSessionId.set('runtime-1')
    $awaitingResponse.set(false)
    $busy.set(false)
    $contextSuggestions.set([])
    $currentCwd.set('/work')
    $currentModel.set('test-model')
    $currentProvider.set('test-provider')
    $freshDraftReady.set(false)
    $gatewayState.set('closed')
    $messages.set([assistantMessage('assistant-1', 'Stable historical answer')])
    $sessionStates.set({})
    $selectedStoredSessionId.set('stored-1')
    $sessions.set([{ id: 'stored-1', message_count: 1, title: 'Stable chat' } as never])
  })

  afterEach(() => {
    cleanup()
    receiptGate.fallbackPublicationIdentity = 'primary:runtime-1'
    receiptGate.suppressed = false
    receiptGate.suppressedPublicationIdentity = null
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
    $sessionStates.set({})
    $selectedStoredSessionId.set(null)
    $sessions.set([])
  })

  it('does not re-render chat history when an unrelated parent idle tick updates', () => {
    const props = {
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
      onTranscribeAudio: vi.fn()
    }

    const queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false } }
    })

    function ParentTickHarness() {
      const [tick, setTick] = useState(0)

      return (
        <QueryClientProvider client={queryClient}>
          <MemoryRouter initialEntries={['/stored-1']}>
            <button onClick={() => setTick(value => value + 1)} type="button">
              parent tick {tick}
            </button>
            <ChatView {...props} />
          </MemoryRouter>
        </QueryClientProvider>
      )
    }

    render(<ParentTickHarness />)

    expect(screen.getByTestId('thread')).toBeTruthy()
    expect(threadRenderCount.current).toBe(1)

    fireEvent.click(screen.getByRole('button', { name: /parent tick/i }))

    // memo(ChatView) with stable props must absorb the parent's idle tick —
    // the transcript (Thread) must not re-render. This is PR #38470's contract.
    expect(threadRenderCount.current).toBe(1)
  })

  const chatProps = () => ({
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
    onTranscribeAudio: vi.fn()
  })

  const visibilityHarness = (
    visible: boolean,
    props: ReturnType<typeof chatProps>,
    onRender: ProfilerOnRenderCallback,
    id: string
  ) => (
    <PaneVisibleContext.Provider value={visible}>
      <Profiler id={id} onRender={onRender}>
        <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
          <MemoryRouter initialEntries={['/stored-1']}>
            <ChatView {...props} />
          </MemoryRouter>
        </QueryClientProvider>
      </Profiler>
    </PaneVisibleContext.Provider>
  )

  it('publishes the current revision on reveal and catches the transcript up before paint', () => {
    const initialMessages = [assistantMessage('assistant-1', 'Initial answer')]
    const currentMessages = [...initialMessages, assistantMessage('assistant-2', 'Background answer')]
    const visibleMessages = [...currentMessages, assistantMessage('assistant-3', 'Visible update')]
    const initialState = {
      ...createClientSessionState('stored-1', initialMessages),
      resumePublicationRevision: 1
    }
    $sessionStates.set({ 'runtime-1': initialState })

    const props = chatProps()
    const commits: { hidden: boolean; messageCount: number; revision: number }[] = []
    const onRender: ProfilerOnRenderCallback = () => {
      const thread = document.querySelector<HTMLElement>('[data-testid="thread"]')

      if (thread) {
        commits.push({
          hidden: Boolean(thread.closest('[data-pane-hidden]')),
          messageCount: Number(thread.dataset.messageCount),
          revision: Number(thread.dataset.revision)
        })
      }
    }
    const harness = (visible: boolean) => visibilityHarness(visible, props, onRender, 'chat-reveal')
    const { rerender } = render(harness(false))
    const commitsBeforeHiddenUpdate = commits.length

    act(() => {
      $sessionStates.set({
        'runtime-1': { ...initialState, messages: currentMessages, resumePublicationRevision: 7 }
      })
    })
    expect(commits).toHaveLength(commitsBeforeHiddenUpdate)

    const commitsBeforeReveal = commits.length
    rerender(harness(true))

    const revealCommits = commits.slice(commitsBeforeReveal)
    expect(revealCommits[0]).toEqual({ hidden: true, messageCount: 1, revision: 7 })
    expect(revealCommits.every(commit => commit.revision === 7)).toBe(true)
    const visibleRevealCommits = revealCommits.filter(commit => !commit.hidden)
    expect(visibleRevealCommits.length).toBeGreaterThan(0)
    expect(visibleRevealCommits.every(commit => commit.messageCount === 2 && commit.revision === 7)).toBe(true)

    act(() => {
      $sessionStates.set({
        'runtime-1': { ...initialState, messages: visibleMessages, resumePublicationRevision: 8 }
      })
    })
    expect(commits.at(-1)).toEqual({ hidden: false, messageCount: 3, revision: 8 })
  })

  it('waits for a Thread leaf receipt bound to the current revision and semantic chain', () => {
    const initialMessages = [assistantMessage('assistant-1', 'Initial answer')]
    const currentMessages = [...initialMessages, assistantMessage('assistant-2', 'Background answer')]
    const initialState = {
      ...createClientSessionState('stored-1', initialMessages),
      resumePublicationRevision: 1
    }
    $sessionStates.set({ 'runtime-1': initialState })

    const props = chatProps()
    const onRender: ProfilerOnRenderCallback = () => {
      const thread = document.querySelector<HTMLElement>('[data-testid="thread"]')

      if (thread) {
        revealTrace.sequence.push({
          at: revealTrace.counter++,
          kind: 'commit',
          hidden: Boolean(thread.closest('[data-pane-hidden]')),
          messageCount: Number(thread.dataset.messageCount),
          revision: Number(thread.dataset.revision)
        })
      }
    }
    const harness = (visible: boolean) => visibilityHarness(visible, props, onRender, 'chat-reveal-receipt')
    const { rerender } = render(harness(true))
    expect(revealTrace.sequence.some(event => event.kind === 'commit' && !event.hidden)).toBe(true)

    rerender(harness(false))
    const eventsBeforePublication = revealTrace.sequence.length
    act(() => {
      $sessionStates.set({
        'runtime-1': { ...initialState, messages: currentMessages, resumePublicationRevision: 7 }
      })
    })
    expect(revealTrace.sequence).toHaveLength(eventsBeforePublication)

    rerender(harness(true))
    const revealEvents = revealTrace.sequence.slice(eventsBeforePublication)
    const commits = revealEvents.filter(
      (event): event is Extract<TraceEvent, { kind: 'commit' }> => event.kind === 'commit'
    )
    const receipts = revealEvents.filter(
      (event): event is Extract<TraceEvent, { kind: 'receipt' }> => event.kind === 'receipt'
    )
    const visibleCommits = commits.filter(commit => !commit.hidden)
    expect(visibleCommits.every(commit => commit.messageCount === 2 && commit.revision === 7)).toBe(true)

    const matchingReceipts = receipts.filter(
      receipt => receipt.revision === 7 && receipt.chainSignature === 'assistant-1\nassistant-2'
    )
    expect(matchingReceipts.length).toBeGreaterThan(0)
    expect(visibleCommits[0]?.at ?? -1).toBeGreaterThan(matchingReceipts[0]?.at ?? -1)
  })

  it('keeps an already-visible streaming surface visible and never re-hides it', () => {
    const initialMessages = [assistantMessage('assistant-1', 'Initial answer')]
    const streamingMessages = [...initialMessages, assistantMessage('assistant-2', 'Streaming tail')]
    const initialState = {
      ...createClientSessionState('stored-1', initialMessages),
      resumePublicationRevision: 7
    }
    $sessionStates.set({ 'runtime-1': initialState })

    const props = chatProps()
    const onRender: ProfilerOnRenderCallback = () => {
      const thread = document.querySelector<HTMLElement>('[data-testid="thread"]')

      if (thread) {
        revealTrace.sequence.push({
          at: revealTrace.counter++,
          kind: 'commit',
          hidden: Boolean(thread.closest('[data-pane-hidden]')),
          messageCount: Number(thread.dataset.messageCount),
          revision: Number(thread.dataset.revision)
        })
      }
    }
    const harness = (visible: boolean) => visibilityHarness(visible, props, onRender, 'chat-streaming')
    const { rerender } = render(harness(true))
    const eventsBeforeStreaming = revealTrace.sequence.length

    act(() => {
      $sessionStates.set({
        'runtime-1': { ...initialState, messages: streamingMessages, resumePublicationRevision: 7 }
      })
    })
    const streamingCommits = revealTrace.sequence
      .slice(eventsBeforeStreaming)
      .filter((event): event is Extract<TraceEvent, { kind: 'commit' }> => event.kind === 'commit')
    expect(streamingCommits.length).toBeGreaterThan(0)
    expect(streamingCommits.every(commit => !commit.hidden)).toBe(true)

    rerender(harness(false))
    rerender(harness(true))
    expect([...revealTrace.sequence].reverse().find(event => event.kind === 'commit')).toMatchObject({
      kind: 'commit',
      hidden: false
    })
  })

  it('re-reveals an unchanged hidden transcript without requiring a fresh leaf commit', () => {
    const initialMessages = [assistantMessage('assistant-1', 'Initial answer')]
    const initialState = {
      ...createClientSessionState('stored-1', initialMessages),
      resumePublicationRevision: 1
    }
    $sessionStates.set({ 'runtime-1': initialState })

    const props = chatProps()
    const onRender: ProfilerOnRenderCallback = () => {
      const thread = document.querySelector<HTMLElement>('[data-testid="thread"]')

      if (thread) {
        revealTrace.sequence.push({
          at: revealTrace.counter++,
          kind: 'commit',
          hidden: Boolean(thread.closest('[data-pane-hidden]')),
          messageCount: Number(thread.dataset.messageCount),
          revision: Number(thread.dataset.revision)
        })
      }
    }
    const harness = (visible: boolean) => visibilityHarness(visible, props, onRender, 'chat-round-trip')
    const { rerender } = render(harness(true))

    rerender(harness(false))
    expect([...revealTrace.sequence].reverse().find(event => event.kind === 'commit')).toMatchObject({
      kind: 'commit',
      hidden: true
    })
    rerender(harness(true))
    expect([...revealTrace.sequence].reverse().find(event => event.kind === 'commit')).toMatchObject({
      kind: 'commit',
      hidden: false
    })
  })

  it('hides the first commit of a session switched into a visible primary pane until its own receipt is accepted', () => {
    // All three sessions carry the SAME transcript bytes and revision, so only
    // the runtime/stored identity (plus the timing of the receipt) tells them
    // apart — a stale receipt captured for the previous session must never
    // reveal the switched-to one.
    const identicalMessages = [assistantMessage('session-msg-1', 'Identical historical answer')]
    const sessionState = (stored: string) => ({
      ...createClientSessionState(stored, identicalMessages),
      resumePublicationRevision: 7
    })
    $activeSessionId.set('runtime-A')
    receiptGate.fallbackPublicationIdentity = 'primary:runtime-A'
    receiptGate.suppressedPublicationIdentity = 'primary:runtime-B'
    $selectedStoredSessionId.set('stored-A')
    $sessionStates.set({ 'runtime-A': sessionState('stored-A') })
    $sessions.set([
      { id: 'stored-A', message_count: 1, title: 'Chat A' } as never,
      { id: 'stored-B', message_count: 1, title: 'Chat B' } as never
    ])

    const props = chatProps()
    const onRender: ProfilerOnRenderCallback = () => {
      const thread = document.querySelector<HTMLElement>('[data-testid="thread"]')

      if (thread) {
        revealTrace.sequence.push({
          at: revealTrace.counter++,
          kind: 'commit',
          hidden: Boolean(thread.closest('[data-pane-hidden]')),
          messageCount: Number(thread.dataset.messageCount),
          revision: Number(thread.dataset.revision)
        })
      }
    }

    function SwitchControls() {
      const navigate = useNavigate()

      return (
        <>
          <button onClick={() => navigate('/stored-B')} type="button">
            switch to B
          </button>
          <button onClick={() => navigate('/stored-C')} type="button">
            switch to C
          </button>
          <button onClick={() => navigate('/stored-D')} type="button">
            switch to D
          </button>
          <button onClick={() => navigate('/stored-D2')} type="button">
            switch to D2
          </button>
        </>
      )
    }

    function SessionSwitchHarness({ visible }: { visible: boolean }) {
      return (
        <PaneVisibleContext.Provider value={visible}>
          <Profiler id="chat-session-switch" onRender={onRender}>
            <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
              <MemoryRouter initialEntries={['/stored-A']}>
                <SwitchControls />
                <ChatView {...props} />
              </MemoryRouter>
            </QueryClientProvider>
          </Profiler>
        </PaneVisibleContext.Provider>
      )
    }

    const harness = (visible: boolean) => <SessionSwitchHarness visible={visible} />
    const { rerender } = render(harness(true))

    const mountedCommits = revealTrace.sequence.filter(
      (event): event is Extract<TraceEvent, { kind: 'commit' }> => event.kind === 'commit'
    )
    expect(mountedCommits.some(commit => !commit.hidden)).toBe(true)

    // The pane stays visible the whole time; only the routed/stored/runtime
    // identity moves from A to B. A real sidebar click lands the ROUTE first,
    // with the store and runtime still bound to A — that very first
    // route-only frame must already be hidden, before any of B's state or
    // runtime exists.
    const switchAt = revealTrace.counter

    act(() => {
      fireEvent.click(screen.getByRole('button', { name: 'switch to B' }))
    })

    const routeOnlyCommits = revealTrace.sequence.filter(
      (event): event is Extract<TraceEvent, { kind: 'commit' }> => event.kind === 'commit' && event.at >= switchAt
    )
    expect(routeOnlyCommits.length).toBeGreaterThan(0)
    expect(routeOnlyCommits.at(-1)).toMatchObject({ hidden: true })

    act(() => {
      $sessionStates.set({ 'runtime-A': sessionState('stored-A'), 'runtime-B': sessionState('stored-B') })
      $activeSessionId.set('runtime-B')
      $selectedStoredSessionId.set('stored-B')
    })

    const postSwitch = revealTrace.sequence.filter(event => event.at >= switchAt)
    const bCommits = postSwitch.filter(
      (event): event is Extract<TraceEvent, { kind: 'commit' }> =>
        event.kind === 'commit' && event.messageCount === 1 && event.revision === 7
    )
    // The first commit of B's transcript must be hidden, and an old-runtime
    // receipt with byte-identical revision/id/content/status cannot reveal it.
    expect(bCommits.length).toBeGreaterThan(0)
    expect(bCommits[0]).toMatchObject({ hidden: true })
    expect(bCommits.at(-1)).toMatchObject({ hidden: true })

    receiptGate.suppressedPublicationIdentity = null
    act(() => {
      $sessionStates.set({
        'runtime-A': sessionState('stored-A'),
        'runtime-B': { ...sessionState('stored-B'), messages: [...identicalMessages] }
      })
    })

    const acceptedBEvents = revealTrace.sequence.filter(event => event.at >= switchAt)
    const acceptedBCommits = acceptedBEvents.filter(
      (event): event is Extract<TraceEvent, { kind: 'commit' }> =>
        event.kind === 'commit' && event.messageCount === 1 && event.revision === 7
    )
    const acceptedBReceipts = acceptedBEvents.filter(
      (event): event is Extract<TraceEvent, { kind: 'receipt' }> => event.kind === 'receipt'
    )

    // ...and only a complete receipt for the current session may reveal it.
    expect(acceptedBReceipts.some(receipt => receipt.publicationIdentity === 'primary:runtime-B')).toBe(true)
    const firstVisibleB = acceptedBCommits.find(commit => !commit.hidden)
    const firstCurrentReceipt = acceptedBReceipts.find(receipt => receipt.publicationIdentity === 'primary:runtime-B')
    expect(firstVisibleB?.at ?? -1).toBeGreaterThan(firstCurrentReceipt?.at ?? -1)

    // A receipt captured while arming at B must not reveal C later: hide the
    // pane (a B receipt is recorded and accepted), switch to byte-identical C
    // with receipts suppressed, then reveal again.
    rerender(harness(false))
    receiptGate.suppressed = true
    const hiddenSwitchAt = revealTrace.counter

    act(() => {
      fireEvent.click(screen.getByRole('button', { name: 'switch to C' }))
      $sessionStates.set({
        'runtime-A': sessionState('stored-A'),
        'runtime-B': sessionState('stored-B'),
        'runtime-C': sessionState('stored-C')
      })
      $activeSessionId.set('runtime-C')
      $selectedStoredSessionId.set('stored-C')
    })

    rerender(harness(true))

    const hiddenSwitchCommits = revealTrace.sequence.filter(
      (event): event is Extract<TraceEvent, { kind: 'commit' }> =>
        event.kind === 'commit' && event.at >= hiddenSwitchAt && event.messageCount === 1 && event.revision === 7
    )
    expect(hiddenSwitchCommits.length).toBeGreaterThan(0)
    // With receipts held back, C must still be hidden: the identity switch
    // cleared the recorded receipt, so nothing pre-switch can reveal it.
    expect(hiddenSwitchCommits.at(-1)).toMatchObject({ hidden: true })

    receiptGate.suppressed = false
    act(() => {
      // Republish C's slice with a fresh messages array (same content) so the
      // boundary re-renders and the unsuppressed Thread leaf re-fires its
      // receipt for C. A same-reference publish is a computed no-op here.
      $sessionStates.set({
        'runtime-A': sessionState('stored-A'),
        'runtime-B': sessionState('stored-B'),
        'runtime-C': { ...sessionState('stored-C'), messages: [...identicalMessages] }
      })
    })

    expect([...revealTrace.sequence].reverse().find(event => event.kind === 'commit')).toMatchObject({
      kind: 'commit',
      hidden: false
    })

    // A real sidebar click can land the route AND the stored selection in one
    // batch while the active runtime and its session state still belong to
    // the previous conversation. In that commit routeSessionMismatch is
    // false, so hiding must key off the runtime still being bound to the old
    // conversation — the first D commit must already be hidden.
    const bothTogetherAt = revealTrace.counter

    act(() => {
      fireEvent.click(screen.getByRole('button', { name: 'switch to D' }))
      $selectedStoredSessionId.set('stored-D')
    })

    const bothTogetherCommits = revealTrace.sequence.filter(
      (event): event is Extract<TraceEvent, { kind: 'commit' }> => event.kind === 'commit' && event.at >= bothTogetherAt
    )
    expect(bothTogetherCommits.length).toBeGreaterThan(0)
    expect(bothTogetherCommits.at(-1)).toMatchObject({ hidden: true })

    // D's runtime binds with its own transcript; only then does a receipt
    // reveal the surface.
    act(() => {
      $sessionStates.set({
        'runtime-A': sessionState('stored-A'),
        'runtime-B': sessionState('stored-B'),
        'runtime-C': sessionState('stored-C'),
        'runtime-D': sessionState('stored-D')
      })
      $activeSessionId.set('runtime-D')
    })

    expect([...revealTrace.sequence].reverse().find(event => event.kind === 'commit')).toMatchObject({
      kind: 'commit',
      hidden: false
    })

    // Auto-compression rotates the stored tip AND the route within one
    // lineage while the runtime keeps its binding — the state slice rotates
    // in place. The surface must stay visible even while the new tip row is
    // still missing from the session list (the lineage key falls back to the
    // raw tip id and changes).
    const compressionAt = revealTrace.counter

    act(() => {
      $sessionStates.set({
        'runtime-A': sessionState('stored-A'),
        'runtime-B': sessionState('stored-B'),
        'runtime-C': sessionState('stored-C'),
        'runtime-D': { ...sessionState('stored-D'), storedSessionId: 'stored-D2' }
      })
      $selectedStoredSessionId.set('stored-D2')
      fireEvent.click(screen.getByRole('button', { name: 'switch to D2' }))
    })

    const compressionCommits = revealTrace.sequence.filter(
      (event): event is Extract<TraceEvent, { kind: 'commit' }> => event.kind === 'commit' && event.at >= compressionAt
    )
    expect(compressionCommits.length).toBeGreaterThan(0)
    expect(compressionCommits.at(-1)).toMatchObject({ hidden: false })

    // The new tip row arrives with its lineage root: the lineage key moves
    // onto the root without re-hiding the surface.
    const rowArrivalAt = revealTrace.counter

    act(() => {
      $sessions.set([
        { id: 'stored-A', message_count: 1, title: 'Chat A' } as never,
        { id: 'stored-B', message_count: 1, title: 'Chat B' } as never,
        { id: 'stored-D2', _lineage_root_id: 'stored-D', message_count: 1, title: 'Chat D' } as never
      ])
    })

    const rowArrivalCommits = revealTrace.sequence.filter(
      (event): event is Extract<TraceEvent, { kind: 'commit' }> => event.kind === 'commit' && event.at >= rowArrivalAt
    )
    expect(rowArrivalCommits.length).toBeGreaterThan(0)
    expect(rowArrivalCommits.at(-1)).toMatchObject({ hidden: false })
  })

  it('reveals a hidden tile from its own receipt while the primary is mid-switch and unbound', () => {
    // Put the GLOBAL primary atoms into the route+selection/runtime-lag
    // state: the primary's runtime is still bound to A while the selection
    // already names B, so the primary binding discriminator is false.
    $activeSessionId.set('runtime-A')
    $selectedStoredSessionId.set('stored-B')
    $sessionStates.set({
      'runtime-A': { ...createClientSessionState('stored-A', [assistantMessage('assistant-1', 'Primary answer')]) }
    })

    const tileMessages = [assistantMessage('tile-msg-1', 'Tile answer')]
    const tileView: SessionView = {
      kind: 'tile',
      $awaitingResponse: atom(false),
      $busy: atom(false),
      $cwd: atom('/work'),
      $fast: atom(false),
      $lastVisibleIsUser: atom(false),
      $messages: atom<ChatMessage[]>(tileMessages),
      $messagesEmpty: atom(false),
      $model: atom('test-model'),
      $provider: atom('test-provider'),
      $reasoningEffort: atom(''),
      $resumePublicationRevision: atom(7),
      $runtimeId: atom('runtime-tile'),
      $storedId: atom('stored-tile')
    }

    const props = chatProps()
    const onRender: ProfilerOnRenderCallback = () => {
      const thread = document.querySelector<HTMLElement>('[data-testid="thread"]')

      if (thread) {
        revealTrace.sequence.push({
          at: revealTrace.counter++,
          kind: 'commit',
          hidden: Boolean(thread.closest('[data-pane-hidden]')),
          messageCount: Number(thread.dataset.messageCount),
          revision: Number(thread.dataset.revision)
        })
      }
    }

    const harness = (visible: boolean) => (
      <SessionViewProvider value={tileView}>
        <PaneVisibleContext.Provider value={visible}>
          <Profiler id="chat-tile-reveal" onRender={onRender}>
            <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
              <MemoryRouter initialEntries={['/stored-1']}>
                <ChatView {...props} />
              </MemoryRouter>
            </QueryClientProvider>
          </Profiler>
        </PaneVisibleContext.Provider>
      </SessionViewProvider>
    )

    // Mount the tile hidden so its reveal gate arms, then reveal it. The
    // tile holds its own transcript and revision — its own receipt is valid —
    // so the primary's unbound state must not keep it hidden.
    const { rerender } = render(harness(false))
    rerender(harness(true))

    expect([...revealTrace.sequence].reverse().find(event => event.kind === 'commit')).toMatchObject({
      kind: 'commit',
      hidden: false
    })
  })

  it('binds reveal matching to runtime publication identity, semantic chain, revision, and completeness', () => {
    const threadAssistant = (id: string, text: string): ThreadMessage =>
      ({
        id,
        role: 'assistant',
        content: [{ type: 'text', text }],
        status: { type: 'complete', reason: 'stop' },
        createdAt: new Date(0),
        metadata: {
          unstable_state: null,
          unstable_annotations: [],
          unstable_data: [],
          steps: [],
          custom: {}
        }
      }) as ThreadMessage
    const firstMessage = threadAssistant('assistant-1', 'First answer')
    const headMessage = threadAssistant('assistant-2', 'Settled answer')
    const contentSignature = commitReceiptContentSignature([firstMessage, headMessage])
    const expectation = {
      chainSignature: 'assistant-1\nassistant-2',
      headMessage,
      contentSignature,
      publicationIdentity: 'primary:runtime-B'
    } as RevealExpectation & { publicationIdentity: string }
    const receipt = {
      revision: 8,
      chainSignature: expectation.chainSignature,
      headMessage,
      contentSignature,
      publicationIdentity: expectation.publicationIdentity,
      complete: true
    } as ThreadCommitReceipt & { publicationIdentity: string }

    expect(revealMatchesExpectation(receipt, expectation, 8)).toBe(true)
    expect(revealMatchesExpectation({ ...receipt, publicationIdentity: 'primary:runtime-A' }, expectation, 8)).toBe(
      false
    )
    expect(revealMatchesExpectation({ ...receipt, revision: 7 }, expectation, 8)).toBe(false)
    expect(revealMatchesExpectation({ ...receipt, chainSignature: 'assistant-1' }, expectation, 8)).toBe(false)
    expect(
      revealMatchesExpectation(
        {
          ...receipt,
          contentSignature: commitReceiptContentSignature([
            threadAssistant('assistant-1', 'Obsolete first answer'),
            headMessage
          ])
        },
        expectation,
        8
      )
    ).toBe(false)
    expect(revealMatchesExpectation({ ...receipt, complete: false }, expectation, 8)).toBe(false)

    const oldUser = {
      id: 'user-same',
      role: 'user',
      content: [{ type: 'text', text: 'Same prompt' }],
      attachments: [{ id: 'old-attachment', type: 'file', name: 'old.txt' }],
      createdAt: new Date(0),
      metadata: { custom: { source: 'runtime-A' } }
    } as unknown as ThreadMessage
    const newUser = {
      ...oldUser,
      attachments: [{ id: 'new-attachment', type: 'file', name: 'new.txt' }],
      metadata: { custom: { source: 'runtime-B' } }
    } as unknown as ThreadMessage
    const metadataBlindSignature = commitReceiptContentSignature([oldUser])

    expect(metadataBlindSignature).toBe(commitReceiptContentSignature([newUser]))
    expect(
      revealMatchesExpectation(
        {
          ...receipt,
          chainSignature: oldUser.id,
          contentSignature: metadataBlindSignature,
          headMessage: oldUser,
          publicationIdentity: 'primary:runtime-A'
        },
        {
          chainSignature: newUser.id,
          contentSignature: commitReceiptContentSignature([newUser]),
          headMessage: newUser,
          publicationIdentity: 'primary:runtime-B'
        } as RevealExpectation & { publicationIdentity: string },
        8
      )
    ).toBe(false)
  })
})
