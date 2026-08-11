import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import type { ThreadMessage } from '@assistant-ui/react'
import { Profiler, type ProfilerOnRenderCallback, useState } from 'react'
import { MemoryRouter } from 'react-router'
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

const threadRenderCount = vi.hoisted(() => ({ current: 0 }))
const revealTrace = vi.hoisted(() => ({ counter: 0, sequence: [] as TraceEvent[] }))

type TraceEvent =
  | { at: number; hidden: boolean; kind: 'commit'; messageCount: number; revision: number }
  | { at: number; chainSignature: string; headMessageId: string | null; kind: 'receipt'; revision: number }

vi.mock('@/components/assistant-ui/thread', async () => {
  const React = await import('react')
  const { useLayoutEffect } = await import('react')
  const { useThread } = await import('@assistant-ui/react')

  return {
    Thread: (props: {
      onCommitReceipt?: (receipt: ThreadCommitReceipt) => void
      resumePublicationRevision?: number
    }) => {
      threadRenderCount.current += 1
      const { messages } = useThread()
      const committedMessages = messages.filter(message => message.metadata?.isOptimistic !== true)

      useLayoutEffect(() => {
        if (!props.onCommitReceipt) {
          return
        }

        const receipt: ThreadCommitReceipt = {
          revision: props.resumePublicationRevision ?? 0,
          chainSignature: committedMessages.map(message => message.id).join('\n'),
          headMessage: committedMessages.at(-1) ?? null,
          contentSignature: commitReceiptContentSignature(committedMessages),
          complete: true
        }

        revealTrace.sequence.push({
          at: revealTrace.counter++,
          kind: 'receipt',
          revision: receipt.revision,
          chainSignature: receipt.chainSignature,
          headMessageId: receipt.headMessage?.id ?? null
        })
        props.onCommitReceipt(receipt)
      })

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

  it('binds reveal matching to the complete semantic chain, revision, and completeness', () => {
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
    const expectation: RevealExpectation = {
      chainSignature: 'assistant-1\nassistant-2',
      headMessage,
      contentSignature
    }
    const receipt: ThreadCommitReceipt = {
      revision: 8,
      chainSignature: expectation.chainSignature,
      headMessage,
      contentSignature,
      complete: true
    }

    expect(revealMatchesExpectation(receipt, expectation, 8)).toBe(true)
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
  })
})
