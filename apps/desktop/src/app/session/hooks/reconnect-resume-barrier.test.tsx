import { QueryClient } from '@tanstack/react-query'
import { act, cleanup, renderHook } from '@testing-library/react'
import { useRef } from 'react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { getLatestSessionMessages } from '@/hermes'
import { type ChatMessage, chatMessageText, toChatMessages } from '@/lib/chat-messages'
import { resetInFlightTurnJournalStateForTests } from '@/lib/inflight-turn-journal'
import { $activeGatewayProfile } from '@/store/profile'
import {
  $messages,
  _resetSessionOwnerHintsForTests,
  setActiveSessionId,
  setAwaitingResponse,
  setBusy,
  setMessages,
  setSelectedStoredSessionId,
  setSessions
} from '@/store/session'
import { clearAllSessionStates } from '@/store/session-states'
import type { SessionMessage, SessionResumeResult } from '@/types/hermes'

import { handleMessageStreamEvent } from './use-message-stream/gateway-event/message-stream'
import type { GatewayEventContext } from './use-message-stream/gateway-event/types'
import { useSessionActions } from './use-session-actions'
import { useSessionStateCache } from './use-session-state-cache'

vi.mock('@/hermes', async original => ({
  ...(await original<Record<string, unknown>>()),
  getLatestSessionMessages: vi.fn()
}))
// A watch window skips the REST prefetch and takes the resume's own snapshot.
const windowKind = vi.hoisted(() => ({ watch: false }))

vi.mock('@/store/windows', async original => ({
  ...(await original<Record<string, unknown>>()),
  isWatchWindow: () => windowKind.watch
}))
vi.mock('@/store/session-control', () => ({ refreshSupportedSessionControlAfterTurn: vi.fn(async () => undefined) }))
vi.mock('@/store/profile', async original => ({
  ...(await original<Record<string, unknown>>()),
  ensureGatewayProfile: vi.fn().mockResolvedValue(undefined)
}))

// The socket dropped while the answer after a tool streamed. The reconnect
// re-resumes cold: its REST page is read before `session.resume` rebinds the
// socket, the turn commits in between, and the tail and message.complete go to
// the dropped socket. Only the idle resume answer says the reply is durable.
const storedId = 'drop-stored'
const runtimeId = 'drop-runtime'

const call = { id: 'call-d', type: 'function', function: { name: 'terminal', arguments: '{}' } }

const beforeCommit: SessionMessage[] = [
  { id: 1, role: 'user', content: 'U1 warm up', timestamp: 10 },
  { id: 2, role: 'assistant', content: 'A1 first answer', timestamp: 11 },
  { id: 3, role: 'user', content: 'U2 across a drop', timestamp: 20 },
  { id: 4, role: 'assistant', content: 'A2i narrating', tool_calls: [call], timestamp: 21 },
  { id: 5, role: 'tool', content: 'core-d', tool_call_id: 'call-d', timestamp: 22 }
] as SessionMessage[]

const committed: SessionMessage[] = [
  ...beforeCommit,
  { id: 6, role: 'assistant', content: 'A2 answer across the drop', timestamp: 23 }
]

const streamingView: ChatMessage[] = [
  ...toChatMessages(beforeCommit.slice(0, 3)),
  {
    id: 'assistant-stream-d',
    role: 'assistant',
    parts: [
      { type: 'text', text: 'A2i narrating' },
      { type: 'tool-call', toolCallId: 'call-d', toolName: 'terminal', result: 'core-d' },
      { type: 'text', text: 'A2 ' }
    ],
    pending: true
  } as ChatMessage
]

const resumedIdle = {
  session_id: runtimeId,
  session_key: storedId,
  resumed: storedId,
  messages: [],
  messages_omitted: true,
  message_count: committed.length,
  running: false,
  info: {}
} as SessionResumeResult

// `beforeResumeAnswers` runs while `session.resume` is in flight, with the
// hook's own view binding (null for the whole of a cold resume).
function mount(beforeResumeAnswers?: (activeSessionIdRef: { current: null | string }) => void) {
  let viewBinding: { current: null | string } = { current: null }

  const requestGateway = vi.fn(async (method: string) => {
    if (method !== 'session.resume') {
      return {}
    }

    beforeResumeAnswers?.(viewBinding)

    return resumedIdle
  })

  return renderHook(() => {
    const busyRef = useRef(false)
    useRef(new QueryClient())

    const cache = useSessionStateCache({
      activeSessionId: null,
      selectedStoredSessionId: storedId,
      busyRef,
      setMessages,
      setBusy,
      setAwaitingResponse
    })

    viewBinding = cache.activeSessionIdRef

    return useSessionActions({
      ...cache,
      activeSessionId: null,
      selectedStoredSessionId: storedId,
      busyRef,
      creatingSessionRef: useRef(false),
      getRouteToken: () => 'drop',
      getRoutedStoredSessionId: () => storedId,
      navigate: vi.fn(),
      requestGateway: requestGateway as never,
      routedSessionId: storedId
    })
  })
}

// The turn's message.complete as the new socket's event pump hands it over,
// read against the view binding the hook really has at that moment.
function completeOnNewSocket(activeSessionIdRef: { current: null | string }): void {
  handleMessageStreamEvent({
    deps: {
      activeGatewayProfile: 'default',
      activeSessionIdRef,
      appendAssistantDelta: vi.fn(),
      appendReasoningDelta: vi.fn(),
      compactedTurnRef: { current: new Set() },
      completeAssistantMessage: vi.fn(),
      failAssistantMessage: vi.fn(),
      finalizeInterimAssistantMessage: vi.fn(),
      flushQueuedDeltas: vi.fn(),
      dropQueuedDeltas: vi.fn(),
      hydrateFromStoredSession: vi.fn(async () => undefined),
      lastCwdInfoSessionRef: { current: null },
      nativeSubagentSessionsRef: { current: new Set() },
      queryClient: {} as GatewayEventContext['deps']['queryClient'],
      refreshHermesConfig: vi.fn(async () => undefined),
      scheduleSessionsRefresh: vi.fn(),
      sessionInterrupted: vi.fn(() => false),
      sessionStateByRuntimeIdRef: { current: new Map() },
      updateSessionState: vi.fn(),
      upsertToolCall: vi.fn()
    },
    event: { type: 'message.complete' },
    explicitSid: runtimeId,
    fromActiveSource: () => true,
    isActiveEvent: false,
    occurredAt: Date.now() / 1000,
    payload: { text: 'A2 answer across the drop', status: 'complete' },
    scheduleConfigRefresh: vi.fn(),
    sessionId: runtimeId
  })
}

const answers = (messages: ChatMessage[]) =>
  messages.filter(message => message.role === 'assistant').map(message => chatMessageText(message).replace(/\s+/g, ''))

beforeEach(() => {
  localStorage.clear()
  clearAllSessionStates()
  resetInFlightTurnJournalStateForTests()
  _resetSessionOwnerHintsForTests()
  $activeGatewayProfile.set('default')
  setActiveSessionId(null)
  setSelectedStoredSessionId(storedId)
  setMessages(streamingView)
  setSessions([
    {
      id: storedId,
      title: storedId,
      source: 'desktop',
      message_count: beforeCommit.length,
      tool_call_count: 1,
      is_active: true,
      started_at: 1,
      last_active: 1,
      ended_at: null,
      model: null,
      preview: null,
      input_tokens: 0,
      output_tokens: 0
    }
  ])
  vi.mocked(getLatestSessionMessages).mockReset()
})

afterEach(() => {
  cleanup()
  clearAllSessionStates()
  resetInFlightTurnJournalStateForTests()
  localStorage.clear()
  setSessions([])
  setMessages([])
  setActiveSessionId(null)
  setSelectedStoredSessionId(null)
  vi.restoreAllMocks()
})

it('shows the committed reply when the turn finishes between the reconnect page read and the resume', async () => {
  vi.mocked(getLatestSessionMessages)
    .mockResolvedValueOnce({ session_id: storedId, messages: beforeCommit })
    .mockResolvedValue({ session_id: storedId, messages: committed })

  const hook = mount()

  await act(async () => {
    await hook.result.current.resumeSession(storedId, true)
  })

  expect(answers($messages.get())).toEqual(['A1firstanswer', 'A2inarratingA2answeracrossthedrop'])
})

// The complete reaches the NEW socket this time, while the cold resume has the
// view unbound: it settles the runtime but never paints. The view still holds
// the first chunk, so the re-read page must win, not that stale view.
it('shows the whole reply when its message.complete lands while the cold resume has the view unbound', async () => {
  vi.mocked(getLatestSessionMessages)
    .mockResolvedValueOnce({ session_id: storedId, messages: beforeCommit })
    .mockResolvedValue({ session_id: storedId, messages: committed })

  const hook = mount(viewBinding => {
    expect(viewBinding.current).toBeNull()
    completeOnNewSocket(viewBinding)
  })

  await act(async () => {
    await hook.result.current.resumeSession(storedId, true)
  })

  expect(answers($messages.get())).toEqual(['A1firstanswer', 'A2inarratingA2answeracrossthedrop'])
})

it('shows the whole reply in a watch window when its message.complete lands while the view is unbound', async () => {
  windowKind.watch = true
  Object.assign(resumedIdle, { messages: committed, messages_omitted: false })

  try {
    const hook = mount(viewBinding => {
      expect(viewBinding.current).toBeNull()
      completeOnNewSocket(viewBinding)
    })

    await act(async () => {
      await hook.result.current.resumeSession(storedId, true)
    })

    expect(getLatestSessionMessages).not.toHaveBeenCalled()
    expect(answers($messages.get())).toEqual(['A1firstanswer', 'A2inarratingA2answeracrossthedrop'])
  } finally {
    windowKind.watch = false
    Object.assign(resumedIdle, { messages: [], messages_omitted: true })
  }
})

it('reads the page once when the resume still reports the turn running', async () => {
  vi.mocked(getLatestSessionMessages).mockResolvedValue({ session_id: storedId, messages: beforeCommit })
  resumedIdle.running = true

  try {
    const hook = mount()

    await act(async () => {
      await hook.result.current.resumeSession(storedId, true)
    })

    expect(getLatestSessionMessages).toHaveBeenCalledTimes(1)
  } finally {
    resumedIdle.running = false
  }
})

// The authority pushes the missed tail and message.complete before it answers
// the resume, and that answer still says running with the committed row in its
// snapshot. The page read before the commit must not win over it.
it('paints the resume snapshot over an older prefetch page while the turn still reports running', async () => {
  vi.mocked(getLatestSessionMessages).mockResolvedValue({ session_id: storedId, messages: beforeCommit })
  Object.assign(resumedIdle, { running: true, messages: committed, messages_omitted: false })

  try {
    const hook = mount()

    await act(async () => {
      await hook.result.current.resumeSession(storedId, true)
    })

    expect(answers($messages.get())).toEqual(['A1firstanswer', 'A2inarratingA2answeracrossthedrop'])
  } finally {
    Object.assign(resumedIdle, { running: false, messages: [], messages_omitted: true })
  }
})
