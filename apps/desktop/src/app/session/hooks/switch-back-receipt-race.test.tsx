import type { GatewayEvent } from '@hermes/shared'
import { QueryClient } from '@tanstack/react-query'
import { act, cleanup, renderHook, waitFor } from '@testing-library/react'
import { useRef } from 'react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { getLatestSessionMessages } from '@/hermes'
import { type ChatMessage, chatMessageText, toChatMessages } from '@/lib/chat-messages'
import { resetInFlightTurnJournalStateForTests } from '@/lib/inflight-turn-journal'
import { $activeGatewayProfile } from '@/store/profile'
import {
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

import { useMessageStream } from './use-message-stream'
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

// Session B streams, the user looks at another chat, then returns to B. The
// session authority writes B's prompt row only when the turn runs, so the
// optimistic prompt learns its stored row from the completion receipt.
const storedId = 'switch-back-stored'
const runtimeId = 'switch-back-runtime'
const submissionId = 'sub-b'
const noop = async () => undefined

const storedRows: SessionMessage[] = [
  { id: 1, role: 'user', content: 'U1 a', timestamp: 10 },
  { id: 2, role: 'assistant', content: 'A1 one', timestamp: 11 },
  { id: 3, role: 'user', content: 'U2 b', timestamp: 20 },
  { id: 4, role: 'assistant', content: 'A2 finished while away', timestamp: 22 }
]

const streaming: ChatMessage[] = [
  ...toChatMessages(storedRows.slice(0, 2)),
  { id: `user-${submissionId}`, role: 'user', parts: [{ type: 'text', text: 'U2 b' }] },
  { id: 'assistant-stream-b', role: 'assistant', parts: [{ type: 'text', text: 'A2 ' }], pending: true }
]

const activated: SessionResumeResult = {
  session_id: runtimeId,
  session_key: storedId,
  resumed: storedId,
  messages: [],
  messages_omitted: true,
  message_count: 3,
  running: true,
  turn_started_at: 19,
  inflight: { user: 'U2 b', assistant: 'A2 ', streaming: true },
  info: {}
} as SessionResumeResult

const completion: GatewayEvent = {
  session_id: runtimeId,
  type: 'message.complete',
  payload: {
    text: 'A2 finished while away',
    status: 'complete',
    persisted_turn: {
      row_ids: [3, 4],
      complete: true,
      user_row_id: 3,
      user_row_ids: [3],
      final_assistant_row_id: 4,
      submission_id: submissionId
    }
  }
} as GatewayEvent

function mount() {
  const requestGateway = vi.fn(async (method: string) => (method === 'session.activate' ? activated : {}))

  const hook = renderHook(() => {
    const busyRef = useRef(false)
    const queryClient = useRef(new QueryClient()).current

    const cache = useSessionStateCache({
      activeSessionId: null,
      selectedStoredSessionId: null,
      busyRef,
      setMessages,
      setBusy,
      setAwaitingResponse
    })

    const actions = useSessionActions({
      ...cache,
      activeSessionId: null,
      selectedStoredSessionId: null,
      busyRef,
      creatingSessionRef: useRef(false),
      getRouteToken: () => 'B',
      getRoutedStoredSessionId: () => storedId,
      navigate: vi.fn(),
      requestGateway: requestGateway as never,
      routedSessionId: null
    })

    const stream = useMessageStream({
      ...cache,
      queryClient,
      hydrateFromStoredSession: noop,
      refreshHermesConfig: noop,
      refreshSessions: noop
    })

    return { cache, actions, stream }
  })

  // Warm cache: B's turn is mid-stream from this window's own send.
  act(() => {
    hook.result.current.cache.updateSessionState(
      runtimeId,
      state => ({
        ...state,
        messages: streaming,
        busy: true,
        awaitingResponse: true,
        turnLive: true,
        sawAssistantPayload: true,
        streamId: 'assistant-stream-b'
      }),
      storedId
    )
  })

  return hook
}

const cachedMessages = (hook: ReturnType<typeof mount>) =>
  hook.result.current.cache.sessionStateByRuntimeIdRef.current.get(runtimeId)!.messages

const occurrences = (messages: ChatMessage[], text: string) =>
  messages.filter(message => chatMessageText(message).trim() === text).length

beforeEach(() => {
  localStorage.clear()
  clearAllSessionStates()
  resetInFlightTurnJournalStateForTests()
  _resetSessionOwnerHintsForTests()
  $activeGatewayProfile.set('default')
  setMessages([])
  setActiveSessionId(null)
  setSelectedStoredSessionId(null)
  setSessions([
    {
      id: storedId,
      title: storedId,
      source: 'desktop',
      message_count: 3,
      tool_call_count: 0,
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

it('paints the prompt and reply once when the receipt binds them while the switch-back page read is in flight', async () => {
  let releasePage!: (page: { session_id: string; messages: SessionMessage[] }) => void
  vi.mocked(getLatestSessionMessages).mockReturnValue(new Promise(resolve => (releasePage = resolve)))
  const hook = mount()

  let pending!: Promise<void>
  await act(async () => {
    pending = hook.result.current.actions.resumeSession(storedId, true)
  })
  await waitFor(() => expect(getLatestSessionMessages).toHaveBeenCalledTimes(1))

  // The warm activate captured the cache before this receipt stamped the
  // optimistic prompt with stored row 3 and settled the stream on row 4.
  act(() => hook.result.current.stream.handleGatewayEvent(completion))
  expect(cachedMessages(hook).find(message => message.id === `user-${submissionId}`)?.rowId).toBe(3)

  await act(async () => {
    releasePage({ session_id: storedId, messages: storedRows })
    await pending
  })

  const messages = cachedMessages(hook)
  expect(occurrences(messages, 'U2 b')).toBe(1)
  expect(occurrences(messages, 'A2 finished while away')).toBe(1)
  expect(messages.map(message => message.rowId)).toEqual([1, 2, 3, 4])
})

it('keeps the receipt binding of a turn that completes while its session is not in the window', async () => {
  vi.mocked(getLatestSessionMessages).mockResolvedValue({ session_id: storedId, messages: storedRows })
  const hook = mount()

  // Another chat owns the window; B's completion still lands in B's cache.
  hook.result.current.cache.activeSessionIdRef.current = 'other-runtime'
  act(() => hook.result.current.stream.handleGatewayEvent(completion))
  expect(cachedMessages(hook).find(message => message.id === `user-${submissionId}`)?.rowId).toBe(3)

  await act(async () => {
    await hook.result.current.actions.resumeSession(storedId, true)
  })

  const messages = cachedMessages(hook)
  expect(occurrences(messages, 'U2 b')).toBe(1)
  expect(occurrences(messages, 'A2 finished while away')).toBe(1)
})
