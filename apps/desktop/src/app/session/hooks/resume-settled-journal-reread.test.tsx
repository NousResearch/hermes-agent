import { QueryClient } from '@tanstack/react-query'
import { act, cleanup, renderHook } from '@testing-library/react'
import { useRef } from 'react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { getLatestSessionMessages } from '@/hermes'
import { chatMessageText } from '@/lib/chat-messages'
import {
  persistInFlightTurnState,
  readInFlightTurnJournal,
  resetInFlightTurnJournalStateForTests
} from '@/lib/inflight-turn-journal'
import { $activeGatewayProfile } from '@/store/profile'
import {
  _resetSessionOwnerHintsForTests,
  setActiveSessionId,
  setAwaitingResponse,
  setBusy,
  setConnection,
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

// A tab died mid-turn and its journal stayed behind. The phone reopens the chat
// just as that turn ends: the cold resume reads the newest REST page while the
// reply is still uncommitted, and the owner's idle answer (no history, nothing
// pending) lands after it. The owner no longer lists the journaled turn, so the
// journal is retired, and the view must not keep the page that predates the
// turn's final reply.
const storedId = 'settled-journal-reread'
const runtimeId = 'settled-journal-runtime'
const submissionId = '5e0c1a2b-7d3f-4c8e-9a1b-2c3d4e5f6a7b'
const noop = async () => undefined

const turnSoFar: SessionMessage[] = [
  { id: 1, role: 'user', content: 'Summarize the open issues.', timestamp: 10 },
  {
    id: 2,
    role: 'assistant',
    content: '',
    timestamp: 11,
    tool_calls: [{ id: 'call-1', function: { name: 'terminal', arguments: '{"command":"ls issues"}' } }]
  },
  { id: 3, role: 'tool', content: 'issue list', tool_call_id: 'call-1', timestamp: 12 }
]

const settledTurn: SessionMessage[] = [
  ...turnSoFar,
  { id: 4, role: 'assistant', content: 'There are four open issues.', timestamp: 13 }
]

function mount(snapshot: SessionResumeResult) {
  const requestGateway = vi.fn().mockResolvedValue(snapshot)

  return renderHook(() => {
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
      getRouteToken: () => 'A',
      getRoutedStoredSessionId: () => null,
      navigate: vi.fn(),
      requestGateway,
      routedSessionId: null
    })

    useMessageStream({
      ...cache,
      queryClient,
      hydrateFromStoredSession: noop,
      refreshHermesConfig: noop,
      refreshSessions: noop
    })

    return { cache, actions }
  })
}

beforeEach(() => {
  vi.useFakeTimers()
  localStorage.clear()
  clearAllSessionStates()
  resetInFlightTurnJournalStateForTests()
  _resetSessionOwnerHintsForTests()
  $activeGatewayProfile.set('default')
  setConnection(null)
  setMessages([])
  setActiveSessionId(null)
  setSelectedStoredSessionId(null)
  setBusy(false)
  setAwaitingResponse(false)
  setSessions([
    {
      id: storedId,
      title: storedId,
      source: 'desktop',
      message_count: 4,
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
  vi.useRealTimers()
  vi.restoreAllMocks()
})

/** The dead tab's journal: its sent prompt and the reply streamed so far. */
async function journalDeadTab() {
  persistInFlightTurnState({
    awaitingResponse: false,
    busy: true,
    messages: [
      { id: `user-${submissionId}`, role: 'user', parts: [{ type: 'text', text: 'Summarize the open issues.' }] },
      {
        id: 'assistant-stream-1',
        role: 'assistant',
        parts: [{ type: 'text', text: 'Reading the issue ' }],
        pending: true
      }
    ],
    storedSessionId: storedId,
    streamId: 'assistant-stream-1',
    turnStartedAt: 10_000
  })
  await vi.advanceTimersByTimeAsync(500)
}

const nextPrompt: SessionMessage = { id: 5, role: 'user', content: 'Now group them.', timestamp: 14 }

it.each([
  ['the owner is idle', false, [], settledTurn],
  [
    'the next queued turn already runs',
    true,
    [{ admission_id: 'adm-2', input_id: '0b6f3c1e-5d2a-4e8f-9c7b-1a2b3c4d5e6f', status: 'started' }],
    [...settledTurn, nextPrompt]
  ]
] as const)(
  're-reads a page that predates the finished journaled turn before retiring the journal when %s',
  async (_case, running, pending, settledPage) => {
    await journalDeadTab()
    vi.mocked(getLatestSessionMessages)
      .mockResolvedValueOnce({ session_id: storedId, messages: turnSoFar })
      .mockResolvedValue({ session_id: storedId, messages: [...settledPage] })

    const { result } = mount({
      session_id: runtimeId,
      resumed: storedId,
      messages: [],
      messages_omitted: true,
      message_count: settledPage.length,
      running,
      info: { pending_submissions: [...pending] }
    } as SessionResumeResult)

    await act(async () => {
      await result.current.actions.resumeSession(storedId, true)
    })

    const rows = result.current.cache.sessionStateByRuntimeIdRef.current.get(runtimeId)!.messages
    const replies = rows.filter(row => row.role === 'assistant' && chatMessageText(row).trim())

    expect(replies.map(row => chatMessageText(row).trim())).toEqual(['There are four open issues.'])
    expect(readInFlightTurnJournal(storedId)).toBeNull()
  }
)

it('reads the page once when no journaled turn has finished', async () => {
  vi.mocked(getLatestSessionMessages).mockResolvedValue({ session_id: storedId, messages: settledTurn })

  const { result } = mount({
    session_id: runtimeId,
    resumed: storedId,
    messages: [],
    messages_omitted: true,
    message_count: 4,
    running: false,
    info: { pending_submissions: [] }
  } as SessionResumeResult)

  await act(async () => {
    await result.current.actions.resumeSession(storedId, true)
  })

  expect(vi.mocked(getLatestSessionMessages)).toHaveBeenCalledTimes(1)
})
