import { AssistantRuntimeProvider, type ThreadMessage } from '@assistant-ui/react'
import { useStore } from '@nanostores/react'
import { act, cleanup, fireEvent, render, renderHook, waitFor, within } from '@testing-library/react'
import { useMemo, useRef, useState } from 'react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { useComposerSubmit } from '@/app/chat/composer/hooks/use-composer-submit'
import { useRuntimeMessageRepository } from '@/app/chat/runtime-repository'
import { PRIMARY_SESSION_VIEW, SessionViewProvider } from '@/app/chat/session-view'
import {
  advanceSessionTranscriptWindow,
  selectTranscriptWindow,
  type SessionWindowMemo
} from '@/app/chat/transcript-window'
import { renderMessageStream } from '@/app/session/hooks/use-message-stream/test-harness'
import type { ChatMessage } from '@/lib/chat-messages'
import { createClientSessionState } from '@/lib/chat-runtime'
import { useIncrementalExternalStoreRuntime } from '@/lib/incremental-external-store-runtime'
import { $clarifyRequests, clearClarifyRequest } from '@/store/clarify'
import { $gateway } from '@/store/gateway'
import { $activeSessionId } from '@/store/session'
import { $sessionStates } from '@/store/session-states'
import { clearTranscriptViewGates, holdTranscriptView } from '@/store/session-transcript-view'

import { stubThreadEnvironment, stubThreadViewportSize } from '../test-utils'

import { TranscriptWindowProvider } from './transcript-window'

import { Thread } from '.'

vi.mock('@/store/native-notifications', () => ({ dispatchNativeNotification: vi.fn() }))
vi.mock('@/components/assistant-ui/wisdom-candidate-card', () => ({ WisdomCandidateCard: () => null }))
vi.mock('@/components/assistant-ui/wisdom-notice-card', () => ({ WisdomNoticeCard: () => null }))
vi.mock('@/components/wisdom-mediation-card', () => ({ WisdomMediationCard: () => null }))

stubThreadEnvironment()
stubThreadViewportSize()

const SID = 'clarify-runtime-a'

const QUESTION = 'Synthetic clarify navigation check: choose a test-only option.'

const CHOICES = ['Choose the test-only option', 'Cancel the test-only check']
const ARGS = { questions: [{ choices: CHOICES, question: QUESTION }] }
const request = vi.fn().mockResolvedValue({ ok: true, remaining: [] })

function seed(tool = true): ChatMessage[] {
  return [
    { id: 'user', role: 'user', parts: [{ type: 'text', text: 'Run the synthetic check' }] },
    { id: 'commentary', role: 'assistant', parts: [{ type: 'text', text: 'A test-only choice is required.' }] },
    ...(tool
      ? [
          {
            id: 'codex-tool',
            role: 'assistant' as const,
            parts: [
              {
                type: 'tool-call' as const,
                toolCallId: 'call-codex',
                toolName: 'clarify',
                args: ARGS,
                argsText: JSON.stringify(ARGS)
              }
            ]
          }
        ]
      : [])
  ]
}

function FullThread() {
  const view = PRIMARY_SESSION_VIEW
  const messages = useStore(view.$messages)
  const busy = useStore(view.$busy)
  const sessionId = useStore(view.$runtimeId)
  const windows = useRef(new Map<string, SessionWindowMemo>())
  const [windowPages, setWindowPages] = useState(1)

  const windowed = useMemo(
    () => advanceSessionTranscriptWindow(windows.current, sessionId ?? '', messages, windowPages).window,
    [messages, sessionId, windowPages]
  )

  const transcriptWindow = useMemo(
    () => ({ olderAvailable: windowed.windowed, expandWindow: () => setWindowPages(pages => pages + 1) }),
    [windowed.windowed]
  )

  const repository = useRuntimeMessageRepository(windowed.messages)

  const runtime = useIncrementalExternalStoreRuntime<ThreadMessage>({
    messageRepository: repository,
    isRunning: busy,
    onNew: async () => {}
  })

  return (
    <SessionViewProvider value={view}>
      <TranscriptWindowProvider value={transcriptWindow}>
        <AssistantRuntimeProvider runtime={runtime}>
          <Thread sessionId={sessionId} />
        </AssistantRuntimeProvider>
      </TranscriptWindowProvider>
    </SessionViewProvider>
  )
}

function mount(messages = seed(), busy = true) {
  const state = { ...createClientSessionState(), messages, busy }
  const states = new Map([[SID, state]])
  const activeSessionIdRef = { current: SID as string | null }
  $activeSessionId.set(SID)
  $sessionStates.set({ [SID]: state })

  const stream = renderMessageStream(SID, {
    states,
    activeSessionIdRef,
    updateSessionState: (id, updater) => {
      const next = updater(states.get(id) ?? createClientSessionState())
      states.set(id, next)
      $sessionStates.set({ ...$sessionStates.get(), [id]: next })

      return next
    }
  })

  const rendered = render(<FullThread />)

  const event = (type: string, payload: Record<string, unknown>, sessionId = SID) =>
    act(() => stream.handleEvent({ type, payload, session_id: sessionId }))

  const clarify = (sessionId = SID, requestId = 'req-independent') =>
    event('clarify.request', { questions: [{ ...ARGS.questions[0], qid: 'q0' }], request_id: requestId }, sessionId)

  return { ...rendered, stream, event, clarify, activeSessionIdRef }
}

async function visibleCard(container: HTMLElement) {
  await waitFor(() => expect(container.querySelectorAll('[data-clarify-batch="1"]')).toHaveLength(1))
  const card = container.querySelector<HTMLElement>('[data-clarify-batch="1"]')!

  // jsdom can prove DOM/CSS visibility, but cannot prove on-screen geometry.
  for (let node: HTMLElement | null = card; node; node = node.parentElement) {
    expect(node.hidden).toBe(false)
    expect(getComputedStyle(node).display).not.toBe('none')
    expect(getComputedStyle(node).visibility).not.toBe('hidden')
  }

  expect(within(card).getByText(QUESTION)).toBeTruthy()

  for (const choice of CHOICES)
    {expect(
      within(card).getByRole('button', { name: new RegExp(choice.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')) })
    ).toBeTruthy()}

  expect(within(card).getByRole('button', { name: /Confirm and continue/ })).toBeTruthy()

  return card
}

beforeEach(() => {
  clearClarifyRequest()
  request.mockClear()
  $gateway.set({ request } as never)
})
afterEach(() => {
  cleanup()
  clearTranscriptViewGates()
  clearClarifyRequest()
  $sessionStates.set({})
  $activeSessionId.set(null)
  $gateway.set(null)
})

it('renders the hydrated Codex tool-only batch through the real stream, store, repository and Thread, then answers q0', async () => {
  const app = mount()
  app.clarify()
  const card = await visibleCard(app.container)
  expect($clarifyRequests.get()[SID]?.requestId).toBe('req-independent')
  const confirm = within(card).getByRole('button', { name: /Confirm and continue/ }) as HTMLButtonElement
  expect(confirm.disabled).toBe(true)
  fireEvent.click(within(card).getByRole('button', { name: new RegExp(CHOICES[0]) }))
  expect(request).not.toHaveBeenCalled()
  expect(confirm.disabled).toBe(false)
  fireEvent.click(confirm)
  await waitFor(() =>
    expect(request).toHaveBeenCalledExactlyOnceWith('clarify.respond', {
      request_id: 'req-independent',
      question_id: 'q0',
      answer: CHOICES[0]
    })
  )
})

it.each(['before', 'after'])('keeps one answerable card when request arrives %s tool.start', async order => {
  const app = mount(seed(false))
  const start = () => app.event('tool.start', { args: ARGS, name: 'clarify', tool_id: 'call-codex' })

  if (order === 'before') {
    app.clarify()
    start()
  } else {
    start()
    app.clarify()
  }

  await visibleCard(app.container)
})

it('keeps the pending request answerable with turn running=false', async () => {
  const app = mount(seed(), false)
  app.clarify()
  expect(app.stream.state().busy).toBe(false)
  const card = await visibleCard(app.container)
  fireEvent.click(within(card).getByRole('button', { name: new RegExp(CHOICES[1]) }))
  fireEvent.click(within(card).getByRole('button', { name: /Confirm and continue/ }))
  await waitFor(() =>
    expect(request).toHaveBeenCalledExactlyOnceWith('clarify.respond', {
      request_id: 'req-independent',
      question_id: 'q0',
      answer: CHOICES[1]
    })
  )
})

it('parks a background request and answers only its own request after switching back', async () => {
  const app = mount()
  const other = 'clarify-runtime-b'
  act(() => {
    app.activeSessionIdRef.current = other
    $activeSessionId.set(other)
  })
  app.clarify()
  expect(app.container.querySelector('[data-clarify-batch="1"]')).toBeNull()
  expect($clarifyRequests.get()[SID]?.requestId).toBe('req-independent')
  app.clarify(other, 'req-other')
  await visibleCard(app.container)
  act(() => {
    app.activeSessionIdRef.current = SID
    $activeSessionId.set(SID)
  })
  const card = await visibleCard(app.container)
  fireEvent.click(within(card).getByRole('button', { name: new RegExp(CHOICES[0]) }))
  fireEvent.click(within(card).getByRole('button', { name: /Confirm and continue/ }))
  await waitFor(() =>
    expect(request).toHaveBeenCalledExactlyOnceWith('clarify.respond', {
      request_id: 'req-independent',
      question_id: 'q0',
      answer: CHOICES[0]
    })
  )
  expect($clarifyRequests.get()[other]?.requestId).toBe('req-other')
})

it('keeps a new pending card in the visible tail beyond an old completed identical question', async () => {
  const old: ChatMessage = {
    id: 'old-tool',
    role: 'assistant',
    parts: [
      {
        type: 'tool-call',
        toolName: 'clarify',
        toolCallId: 'old-call',
        args: ARGS,
        argsText: JSON.stringify(ARGS),
        result: { responses: [{ question: QUESTION, user_response: CHOICES[1] }] }
      }
    ]
  }

  const history: ChatMessage[] = Array.from({ length: 800 }, (_, i) => ({
    id: `history-${i}`,
    role: i % 2 ? 'assistant' : 'user',
    parts: [{ type: 'text', text: `历史 ${i}` }]
  }))

  const messages = [old, ...history, ...seed()]
  expect(selectTranscriptWindow(messages).windowed).toBe(true)
  expect(selectTranscriptWindow(messages).messages.some(message => message.id === old.id)).toBe(false)
  const app = mount(messages)
  app.clarify()
  const card = await visibleCard(app.container)
  expect(app.stream.state().messages[0]).toBe(old)
  fireEvent.click(within(card).getByRole('button', { name: new RegExp(CHOICES[0]) }))
  fireEvent.click(within(card).getByRole('button', { name: /Confirm and continue/ }))
  await waitFor(() =>
    expect(request).toHaveBeenCalledExactlyOnceWith('clarify.respond', {
      request_id: 'req-independent',
      question_id: 'q0',
      answer: CHOICES[0]
    })
  )
})

it('ordinary composer text skips with an empty answer and never approves the choice', async () => {
  const app = mount(seed(), false)
  app.clarify()
  await visibleCard(app.container)
  const onSubmit = vi.fn().mockResolvedValue(true)

  const { result } = renderHook(() =>
    useComposerSubmit({
      activeQueueSessionKey: SID,
      activeQueueSessionKeyRef: { current: SID },
      attachments: [],
      busy: false,
      compacting: false,
      clearDraft: vi.fn(),
      disabled: false,
      draftRef: { current: '允许' },
      drainNextQueued: vi.fn().mockResolvedValue(false),
      editorRef: { current: null },
      exitQueuedEdit: vi.fn(),
      focusInput: vi.fn(),
      inputDisabled: false,
      loadIntoComposer: vi.fn(),
      onCancel: vi.fn(),
      onSteer: vi.fn(),
      onSubmit,
      queueCurrentDraft: vi.fn(),
      queueEdit: null,
      queuedPrompts: [],
      sessionId: SID,
      setComposerText: vi.fn(),
      stashAt: vi.fn()
    })
  )

  act(() => result.current.submitDraft())
  await waitFor(() =>
    expect(request).toHaveBeenCalledExactlyOnceWith('clarify.respond', { request_id: 'req-independent', answer: '' })
  )
  expect(onSubmit).toHaveBeenCalledWith('允许', { attachments: [], composerScope: SID })
  expect($clarifyRequests.get()[SID]).toBeUndefined()
})

it('expires a held request through the real event path without exposing cached history', async () => {
  holdTranscriptView(SID)
  const app = mount([{ id: 'raw', role: 'assistant', parts: [{ type: 'text', text: 'UNVERIFIED history' }] }])
  app.clarify()
  await visibleCard(app.container)
  expect(app.container.textContent).not.toContain('UNVERIFIED')
  app.event('clarify.expire', { request_id: 'req-independent' })
  await waitFor(() => expect(app.container.querySelectorAll('form[data-clarify-batch]')).toHaveLength(0))
  expect(PRIMARY_SESSION_VIEW.$messages.get()).toEqual([])
  expect(app.container.textContent).not.toContain('UNVERIFIED')
})
