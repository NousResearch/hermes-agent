import { useStore } from '@nanostores/react'
import { QueryClient } from '@tanstack/react-query'
import { act, cleanup, renderHook } from '@testing-library/react'
import { useMemo, useRef } from 'react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { useComposerQueue } from '@/app/chat/composer/hooks/use-composer-queue'
import { useMessageStream } from '@/app/session/hooks/use-message-stream'
import { useSubmitPrompt } from '@/app/session/hooks/use-prompt-actions/submit'
import { useSessionStateCache } from '@/app/session/hooks/use-session-state-cache'
import { setApiRequestConnection, setApiRequestProfile } from '@/hermes'
import { en } from '@/i18n/en'
import { chatMessageText, textPart, toChatMessages } from '@/lib/chat-messages'
import { $queuedPromptsBySession, enqueueQueuedPrompt, getQueuedPrompts } from '@/store/composer-queue'
import {
  $busy,
  $messages,
  $sessions,
  setActiveSessionId,
  setAwaitingResponse,
  setBusy,
  setMessages
} from '@/store/session'
import { $sessionStates, $workingSessionIds, clearAllSessionStates } from '@/store/session-states'
import { $todosBySession, clearSessionTodos, setSessionTodos } from '@/store/todos'
import { transcriptTailState } from '@/store/transcript-tail'
import type { SessionInfo, SessionMessage, SessionMessagesResponse } from '@/types/hermes'

import { createPostTurnHydrator } from './post-turn-hydration'

vi.mock('@/lib/complete-sound', () => ({ playCompleteSound: vi.fn() }))
vi.mock('@/lib/haptics', () => ({ triggerHaptic: vi.fn() }))
vi.mock('@/store/native-notifications', () => ({ dispatchNativeNotification: vi.fn() }))

const SID = 'runtime-queued-hydration'
const STORED = 'stored-queued-hydration'
const prompt = 'Repeat the same prompt'
const reply = 'The same reply'

const sessionRow: SessionInfo = {
  id: STORED,
  ended_at: null,
  input_tokens: 0,
  output_tokens: 0,
  is_active: true,
  last_active: 0,
  message_count: 0,
  model: null,
  preview: null,
  source: 'desktop',
  started_at: 0,
  title: null,
  tool_call_count: 0
}

const persistedA: SessionMessage[] = [
  { id: 1, role: 'user', content: prompt, timestamp: 1 },
  { id: 2, role: 'assistant', content: reply, timestamp: 2 }
]

const persistedBoth: SessionMessage[] = [
  ...persistedA,
  { id: 3, role: 'user', content: prompt, timestamp: 3 },
  { id: 4, role: 'assistant', content: reply, timestamp: 4 }
]

function deferred<T>() {
  let resolve!: (value: T) => void

  const promise = new Promise<T>(done => {
    resolve = done
  })

  return { promise, resolve }
}

function mount() {
  const reads: ReturnType<typeof deferred<SessionMessagesResponse>>[] = []

  const api = vi.fn(() => {
    const read = deferred<SessionMessagesResponse>()
    reads.push(read)

    return read.promise
  })

  Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { api } })
  const requestGateway = vi.fn(async () => ({ queued: true }))
  setActiveSessionId(SID)

  const hook = renderHook(
    ({ drain }: { drain: boolean }) => {
      const busyRef = useRef(false)
      const queryClient = useMemo(() => new QueryClient(), [])
      const busy = useStore($busy)

      const cache = useSessionStateCache({
        activeSessionId: SID,
        busyRef,
        selectedStoredSessionId: STORED,
        setAwaitingResponse,
        setBusy,
        setMessages
      })

      const {
        activeSessionIdRef,
        selectedStoredSessionIdRef,
        sessionStateByRuntimeIdRef,
        runtimeIdByStoredSessionIdRef,
        updateSessionState
      } = cache

      const hydrate = useMemo(
        () =>
          createPostTurnHydrator({
            activeSessionIdRef,
            selectedStoredSessionIdRef,
            sessionStateByRuntimeIdRef,
            runtimeIdByStoredSessionIdRef,
            updateSessionState
          }),
        [
          activeSessionIdRef,
          selectedStoredSessionIdRef,
          sessionStateByRuntimeIdRef,
          runtimeIdByStoredSessionIdRef,
          updateSessionState
        ]
      )

      const stream = useMessageStream({
        ...cache,
        hydrateFromStoredSession: hydrate,
        queryClient,
        refreshHermesConfig: async () => undefined,
        refreshSessions: async () => undefined
      })

      const submit = useSubmitPrompt({
        ...cache,
        busyRef,
        copy: en.desktop,
        createBackendSessionForSend: async () => null,
        getRoutedStoredSessionId: () => STORED,
        getRouteToken: () => `/chat/${STORED}`,
        requestGateway: requestGateway as Parameters<typeof useSubmitPrompt>[0]['requestGateway'],
        resumeStoredSession: async () => undefined,
        syncAttachmentsForSubmit: async (sessionId, attachments) => ({ sessionId, attachments })
      })

      useComposerQueue({
        activeQueueSessionKey: STORED,
        queueSessionKey: STORED,
        sessionId: SID,
        attachments: [],
        busy: busy || !drain,
        clearDraft: () => undefined,
        draftRef: useRef(''),
        focusInput: () => undefined,
        loadIntoComposer: () => undefined,
        onCancel: () => undefined,
        onSteer: undefined,
        onSubmit: submit,
        queueEditRef: useRef(null)
      })

      return { cache, hydrate, stream }
    },
    { initialProps: { drain: false } }
  )

  const event = async (type: string, text = '') => {
    await act(async () => hook.result.current.stream.handleGatewayEvent({ type, session_id: SID, payload: { text } }))
  }

  const release = async (index: number, messages = persistedA) => {
    await act(async () => {
      reads[index].resolve({ session_id: STORED, messages })
      await reads[index].promise
    })
  }

  const state = () => hook.result.current.cache.sessionStateByRuntimeIdRef.current.get(SID)!

  return { hook, api, reads, requestGateway, event, release, state }
}

beforeEach(() => {
  window.localStorage.clear()
  clearAllSessionStates()
  $queuedPromptsBySession.set({})
  setBusy(false)
  setMessages([])
  $sessions.set([])
  clearSessionTodos(SID)
  setApiRequestConnection(null)
  setApiRequestProfile(null)
})
afterEach(() => {
  cleanup()
  clearAllSessionStates()
  $queuedPromptsBySession.set({})
  clearSessionTodos(SID)
  setApiRequestConnection(null)
  setApiRequestProfile(null)
  vi.restoreAllMocks()
})

it.each(
  [
    'before-seed',
    'after-seed',
    'during-stream',
    'during-stream-persisted',
    'stream-structure',
    'multiple-queued',
    'after-complete',
    'read-starts-after-seed',
    'newest-first'
  ].flatMap(ordering => [true, false].map(resumedUser => ({ ordering, resumedUser })))
)('keeps the accepted queued turn ($ordering, resumed user=$resumedUser)', async ({ ordering, resumedUser }) => {
  const h = mount()
  await act(async () => {
    h.hook.result.current.cache.updateSessionState(
      SID,
      state => ({
        ...state,
        messages: toChatMessages(resumedUser ? persistedA.slice(0, 1) : []),
        busy: true,
        awaitingResponse: true,
        adoptedRunningTurn: true,
        turnLive: true
      }),
      STORED
    )
    enqueueQueuedPrompt(STORED, { text: prompt, attachments: [] })
  })
  await h.event('message.start')
  await h.event('message.complete', reply)
  expect(h.reads).toHaveLength(1)

  if (ordering === 'before-seed') {
    await h.release(0)
  }

  await act(async () => h.hook.rerender({ drain: true }))
  expect(h.requestGateway).toHaveBeenCalledTimes(1)
  expect(h.requestGateway).toHaveBeenCalledWith(
    'prompt.submit',
    expect.objectContaining({
      session_id: SID,
      text: prompt,
      queued: true
    }),
    expect.any(Number)
  )
  expect(getQueuedPrompts(STORED)).toEqual([])
  let readIndex = 0

  if (ordering === 'read-starts-after-seed') {
    void h.hook.result.current.hydrate(1, STORED, SID)
    readIndex = 1
  }

  if (ordering === 'after-seed' || ordering === 'read-starts-after-seed') {
    await h.release(readIndex)
  }

  await h.event('message.start')
  await h.event('message.delta', reply)

  if (ordering === 'stream-structure') {
    await act(async () => {
      h.hook.result.current.stream.appendReasoningDelta(SID, 'fixture reasoning')
      h.hook.result.current.stream.appendReasoningDelta(SID, 'fixture reasoning', true)
    })
    const streamed = h.state().messages.at(-1)!
    expect(streamed.pending).toBe(true)
    expect(streamed.parts.some(part => part.type === 'reasoning')).toBe(true)
    await h.release(0, persistedBoth)
    expect(h.state().messages.find(message => message.id === streamed.id)).toMatchObject({
      id: streamed.id,
      pending: true,
      parts: expect.arrayContaining([
        expect.objectContaining({ type: 'reasoning', text: 'fixture reasoning' }),
        expect.objectContaining({ type: 'text', text: reply })
      ])
    })
    expect(h.state().streamId).toBe(streamed.id)
  }

  if (ordering === 'during-stream') {
    await h.release(0)
  }

  if (ordering === 'during-stream-persisted') {
    await h.release(0, persistedBoth)
  }

  if (ordering === 'newest-first') {
    void h.hook.result.current.hydrate(1, STORED, SID)
    await h.release(1, persistedBoth)
    await h.release(0)
  }

  if (ordering !== 'after-complete') {
    expect(h.state().busy).toBe(true)
    expect($sessionStates.get()[SID].busy).toBe(true)
    expect($workingSessionIds.get()).toContain(STORED)
  }

  await h.event('message.complete', reply)

  let finalRows = persistedBoth

  if (ordering === 'multiple-queued') {
    await act(async () => {
      enqueueQueuedPrompt(STORED, { text: prompt, attachments: [] })
    })
    expect(h.requestGateway).toHaveBeenCalledTimes(2)
    await h.event('message.start')
    await h.event('message.delta', reply)
    await h.event('message.complete', reply)
    await h.release(0)
    finalRows = [
      ...persistedBoth,
      { id: 5, role: 'user', content: prompt },
      { id: 6, role: 'assistant', content: reply }
    ]
  }

  if (ordering === 'after-complete') {
    await h.release(0)
  }

  const visible = () =>
    $messages
      .get()
      .filter(message => !message.hidden)
      .map(message => [message.role, chatMessageText(message)])

  const expected = finalRows.map(message => [message.role, message.content])
  expect(visible()).toEqual(expected)
  // A later authoritative read must adopt durable rows, not duplicate the
  // identical local turn or retain an obsolete optimistic copy.
  const finalIndex = h.reads.length
  void h.hook.result.current.hydrate(1, STORED, SID)
  await h.release(finalIndex, finalRows)
  expect(visible()).toEqual(expected)
  expect(
    h
      .state()
      .messages.filter(message => message.rowId !== undefined)
      .map(message => message.rowId)
  ).toEqual(finalRows.map(message => message.id))

  if (ordering === 'stream-structure') {
    expect(
      h
        .state()
        .messages.find(message => message.rowId === 4)
        ?.parts.some(part => part.type === 'reasoning')
    ).toBe(true)
  }

  if (ordering === 'read-starts-after-seed') {
    const tail = transcriptTailState(STORED, { connectionId: 'local', profile: 'default' })
    await h.release(0)
    expect(transcriptTailState(STORED, { connectionId: 'local', profile: 'default' })).toBe(tail)
    expect(visible()).toEqual([
      ['user', prompt],
      ['assistant', reply],
      ['user', prompt],
      ['assistant', reply]
    ])
  }

  expect(h.requestGateway).toHaveBeenCalledTimes(ordering === 'multiple-queued' ? 2 : 1)
})

it.each([
  'backfill-structure',
  'authoritative-rewrite',
  'authoritative-rewrite-optimistic',
  'runtime-rebind',
  'stored-rotation',
  'authority-epoch',
  'connection',
  'profile',
  'lineage',
  'route',
  'new-todos',
  'wrong-response',
  'owned-route'
])('publishes history only within its owner and preserves newer work (%s)', async change => {
  const h = mount()
  const oldPrefix = toChatMessages([{ id: 90, role: 'system', content: 'earlier page', timestamp: 0 }])

  const error = {
    id: 'local-error',
    role: 'assistant' as const,
    error: 'local failure',
    parts: [textPart('local failure')]
  }

  const tool = {
    type: 'tool-call' as const,
    toolCallId: 'owned-tool',
    toolName: 'read_file',
    args: {},
    argsText: '{}',
    result: 'kept'
  }

  await act(async () => {
    h.hook.result.current.cache.updateSessionState(
      SID,
      state => ({
        ...state,
        messages: [
          ...oldPrefix,
          ...toChatMessages(persistedA).map(message =>
            message.role === 'assistant' ? { ...message, parts: [...message.parts, tool] } : message
          ),
          error
        ]
      }),
      STORED
    )
  })

  if (change === 'authoritative-rewrite-optimistic') {
    await act(async () =>
      h.hook.result.current.cache.updateSessionState(SID, state => ({
        ...state,
        messages: state.messages.map(message => ({ ...message, rowId: undefined, id: `${message.role}-${message.id}` }))
      }))
    )
  }

  const scope =
    change === 'owned-route'
      ? { connectionId: 'owner-connection', profile: 'owner-profile' }
      : { connectionId: 'local', profile: 'default' }

  if (change === 'owned-route') {
    $sessions.set([{ ...sessionRow, connection_id: scope.connectionId, profile: scope.profile }])
  }

  const pending = h.hook.result.current.hydrate(1, STORED, SID)
  expect(h.api).toHaveBeenCalledWith(expect.objectContaining(scope))
  const cache = h.hook.result.current.cache
  const tail = transcriptTailState(STORED, scope)

  const changes: Record<string, () => void> = {
    'backfill-structure': () => undefined,
    'authoritative-rewrite': () => undefined,
    'authoritative-rewrite-optimistic': () => undefined,
    'runtime-rebind': () => {
      cache.runtimeIdByStoredSessionIdRef.current.set(STORED, 'replacement')
    },
    'stored-rotation': () => {
      cache.updateSessionState(SID, state => state, 'rotated')
    },
    'authority-epoch': () => {
      cache.updateSessionState(SID, state => ({ ...state, transcriptAuthorityEpoch: 1 }))
    },
    connection: () => setApiRequestConnection('other-connection'),
    profile: () => setApiRequestProfile('other-profile'),
    lineage: () => {
      $sessions.set([{ ...sessionRow, _lineage_root_id: 'different-root' }])
    },
    'owned-route': () => {
      setApiRequestConnection('other-ambient')
      setApiRequestProfile('other-ambient')
    },
    route: () => {
      cache.activeSessionIdRef.current = 'foreground-other'
      cache.selectedStoredSessionIdRef.current = 'stored-other'
      setActiveSessionId('foreground-other')
      setMessages([{ id: 'foreground-user', role: 'user', parts: [textPart('other chat')] }])
    },
    'new-todos': () => {
      cache.updateSessionState(SID, state => ({
        ...state,
        busy: true,
        awaitingResponse: true,
        streamId: 'next-stream'
      }))
      setSessionTodos(SID, [{ id: 'next-task', content: 'new work', status: 'in_progress' }])
    },
    'wrong-response': () => undefined
  }

  await act(async () => changes[change]())
  const before = h.state()
  const view = $messages.get()

  const nextRows: SessionMessage[] = change.startsWith('authoritative-rewrite')
    ? [{ id: 100, role: 'system', content: 'authoritative replacement' }]
    : [...persistedA, { id: 5, role: 'system', content: 'authoritative extra' }]

  await act(async () => {
    h.reads[0].resolve({ session_id: change === 'wrong-response' ? 'wrong-owner' : STORED, messages: nextRows })
    await pending
  })

  const rejected = [
    'runtime-rebind',
    'stored-rotation',
    'authority-epoch',
    'connection',
    'profile',
    'lineage',
    'wrong-response'
  ].includes(change)

  if (rejected) {
    expect(h.state()).toBe(before)
    expect($messages.get()).toBe(view)
    expect(transcriptTailState(STORED, scope)).toBe(tail)
  } else if (change.startsWith('authoritative-rewrite')) {
    expect(h.state().messages.map(chatMessageText)).toEqual(['authoritative replacement', 'local failure'])
  } else {
    expect(h.state().messages.map(chatMessageText)).toEqual([
      'earlier page',
      prompt,
      reply,
      'authoritative extra',
      'local failure'
    ])
    expect(h.state().messages.find(message => message.rowId === 2)?.parts).toContainEqual(tool)

    if (change === 'route') {
      expect($messages.get()).toBe(view)
    }

    if (change === 'new-todos') {
      expect(h.state().busy).toBe(true)
      expect(h.state().awaitingResponse).toBe(true)
      expect(h.state().streamId).toBe('next-stream')
      expect($workingSessionIds.get()).toContain(STORED)
      expect($todosBySession.get()[SID]).toEqual([{ id: 'next-task', content: 'new work', status: 'in_progress' }])
    }
  }

  expect(h.requestGateway).not.toHaveBeenCalled()
})
