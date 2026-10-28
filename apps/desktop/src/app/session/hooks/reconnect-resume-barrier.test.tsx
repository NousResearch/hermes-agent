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

import { useSessionActions } from './use-session-actions'
import { useSessionStateCache } from './use-session-state-cache'

vi.mock('@/hermes', async original => ({
  ...(await original<Record<string, unknown>>()),
  getLatestSessionMessages: vi.fn()
}))
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

function mount() {
  const requestGateway = vi.fn(async (method: string) => (method === 'session.resume' ? resumedIdle : {}))

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
