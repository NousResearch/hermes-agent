import type { GatewayEvent } from '@hermes/shared'
import { QueryClient } from '@tanstack/react-query'
import { act, cleanup, renderHook } from '@testing-library/react'
import { useRef } from 'react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { getLatestSessionMessages } from '@/hermes'
import { type ChatMessage, chatMessageText, textPart } from '@/lib/chat-messages'
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
      getRouteToken: () => 'A',
      getRoutedStoredSessionId: () => null,
      navigate: vi.fn(),
      requestGateway,
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

  return Object.assign(hook, { requestGateway })
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
      pending: [...pending]
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

it('keeps folding the journal when the page that would show the finished reply cannot be re-read', async () => {
  await journalDeadTab()
  vi.mocked(getLatestSessionMessages)
    .mockResolvedValueOnce({ session_id: storedId, messages: turnSoFar })
    .mockRejectedValue(new Error('REST unavailable'))

  const { result } = mount({
    session_id: runtimeId,
    resumed: storedId,
    messages: [],
    messages_omitted: true,
    message_count: settledTurn.length,
    running: false,
    pending: []
  } as SessionResumeResult)

  await act(async () => {
    await result.current.actions.resumeSession(storedId, true)
  })

  const rows = result.current.cache.sessionStateByRuntimeIdRef.current.get(runtimeId)!.messages
  const replies = rows.filter(row => row.role === 'assistant' && chatMessageText(row).trim())

  expect(vi.mocked(getLatestSessionMessages)).toHaveBeenCalledTimes(2)
  expect(replies.map(row => chatMessageText(row).trim())).toEqual(['Reading the issue'])
  expect(readInFlightTurnJournal(storedId)).not.toBeNull()
})

it('leaves the running successor its own stream when the finished turn page cannot be re-read', async () => {
  await journalDeadTab()
  vi.mocked(getLatestSessionMessages)
    .mockResolvedValueOnce({ session_id: storedId, messages: turnSoFar })
    .mockRejectedValue(new Error('REST unavailable'))

  const { result } = mount({
    session_id: runtimeId,
    resumed: storedId,
    messages: [],
    messages_omitted: true,
    message_count: settledTurn.length + 1,
    running: true,
    turn_started_at: 30,
    inflight: { user: 'Now group them.', assistant: 'Grouping ', streaming: true },
    pending: [{ admission_id: 'adm-2', input_id: '0b6f3c1e-5d2a-4e8f-9c7b-1a2b3c4d5e6f', status: 'started' }]
  } as SessionResumeResult)

  await act(async () => {
    await result.current.actions.resumeSession(storedId, true)
  })

  act(() =>
    result.current.stream.handleGatewayEvent({
      session_id: runtimeId,
      type: 'message.delta',
      payload: { text: 'them by area' }
    } as GatewayEvent)
  )
  await act(async () => {
    await vi.advanceTimersByTimeAsync(150)
  })

  const state = result.current.cache.sessionStateByRuntimeIdRef.current.get(runtimeId)!
  const replies = state.messages.filter(row => row.role === 'assistant' && chatMessageText(row).trim())

  expect(replies.map(row => chatMessageText(row).trim())).toEqual(['Reading the issue', 'Grouping them by area'])
  expect(replies[0].pending).toBeFalsy()
  expect(state.turnStartedAt).toBe(30_000)
})

/** The newest page after the session moved on from another device; the journaled prompt is older. */
const newerTurns: SessionMessage[] = [
  { id: 30, role: 'user', content: 'Draft a note for the team.', timestamp: 30 },
  { id: 31, role: 'assistant', content: 'Here is the note for the team.', timestamp: 31 }
]

it('retires a finished turn by the owner list alone when its prompt is off the loaded page', async () => {
  await journalDeadTab()
  vi.mocked(getLatestSessionMessages).mockResolvedValue({ session_id: storedId, messages: newerTurns })

  const { result } = mount({
    session_id: runtimeId,
    resumed: storedId,
    messages: [],
    messages_omitted: true,
    message_count: 40,
    running: false,
    pending: []
  } as SessionResumeResult)

  await act(async () => {
    await result.current.actions.resumeSession(storedId, true)
  })

  const rows = result.current.cache.sessionStateByRuntimeIdRef.current.get(runtimeId)!.messages

  expect(rows.map(row => chatMessageText(row).trim())).toEqual([
    'Draft a note for the team.',
    'Here is the note for the team.'
  ])
  expect(readInFlightTurnJournal(storedId)).toBeNull()
})

it('does not retire a journal by a pending list the snapshot did not carry itself', async () => {
  await journalDeadTab()
  vi.mocked(getLatestSessionMessages).mockResolvedValue({ session_id: storedId, messages: newerTurns })

  const { result } = mount({
    session_id: runtimeId,
    resumed: storedId,
    messages: [],
    messages_omitted: true,
    message_count: 40,
    running: false,
    info: { pending_submissions: [] }
  } as SessionResumeResult)

  await act(async () => {
    await result.current.actions.resumeSession(storedId, true)
  })

  expect(readInFlightTurnJournal(storedId)).not.toBeNull()
})

it('reads the page once when the owner still lists the journaled turn', async () => {
  await journalDeadTab()
  vi.mocked(getLatestSessionMessages).mockResolvedValue({ session_id: storedId, messages: turnSoFar })

  const { result } = mount({
    session_id: runtimeId,
    resumed: storedId,
    messages: [],
    messages_omitted: true,
    message_count: turnSoFar.length,
    running: true,
    pending: [{ admission_id: 'adm-1', input_id: submissionId, status: 'started' }]
  } as SessionResumeResult)

  await act(async () => {
    await result.current.actions.resumeSession(storedId, true)
  })

  expect(vi.mocked(getLatestSessionMessages)).toHaveBeenCalledTimes(1)
  expect(readInFlightTurnJournal(storedId)).not.toBeNull()
})

// The journal names the execution that streamed it (every owner frame carries the claim's `admission_id`), not
// the prompt bubble before it: a regenerate keeps the first send's bubble, and a queue drain adds no bubble, so
// the nearest prompt names an admission the owner settled long ago.
const firstSend: ChatMessage = {
  id: `user-${submissionId}`,
  role: 'user',
  parts: [textPart('Summarize the open issues.')],
  rowId: 1
}

const firstReply: ChatMessage = {
  id: 'reply-1',
  role: 'assistant',
  parts: [textPart('There are four open issues.')],
  rowId: 4
}

const execution = (admission: string) => ({ admission_id: admission, authority_epoch: 7, execution_generation: 2 })

/** A live tab streams `text` from the owner's execution of `admission`, then dies; its journal stays behind. */
async function streamThenDie(transcript: ChatMessage[], admission: string, text: string) {
  const live = mount({
    session_id: runtimeId,
    resumed: storedId,
    messages: [],
    message_count: 0
  } as SessionResumeResult)

  act(() => {
    live.result.current.cache.updateSessionState(runtimeId, state => ({ ...state, messages: transcript }), storedId)
  })

  for (const event of [
    { type: 'message.start', payload: {} },
    { type: 'message.delta', payload: { text } }
  ]) {
    act(() =>
      live.result.current.stream.handleGatewayEvent({
        session_id: runtimeId,
        ...execution(admission),
        ...event
      } as GatewayEvent)
    )
  }

  await act(async () => {
    await vi.advanceTimersByTimeAsync(600)
  })
  live.unmount()
  clearAllSessionStates()
  expect(readInFlightTurnJournal(storedId)).not.toBeNull()
}

async function reopen(snapshot: Partial<SessionResumeResult>, page: SessionMessage[]) {
  vi.mocked(getLatestSessionMessages).mockResolvedValue({ session_id: storedId, messages: page })

  const { result } = mount({
    session_id: runtimeId,
    resumed: storedId,
    messages: [],
    messages_omitted: true,
    message_count: page.length,
    running: false,
    ...snapshot
  } as SessionResumeResult)

  await act(async () => {
    await result.current.actions.resumeSession(storedId, true)
  })
  await act(async () => {
    await vi.advanceTimersByTimeAsync(600)
  })

  return result.current.cache.sessionStateByRuntimeIdRef.current
    .get(runtimeId)!
    .messages.filter(row => row.role === 'assistant' && chatMessageText(row).trim())
    .map(row => chatMessageText(row).trim())
}

const resentId = '0b6f3c1e-5d2a-4e8f-9c7b-1a2b3c4d5e6f'

const earlierExchange: SessionMessage[] = [
  { id: 1, role: 'user', content: 'List the repositories.', timestamp: 5 },
  { id: 2, role: 'assistant', content: 'There are two repositories.', timestamp: 6 }
]

const regenerated: ChatMessage[] = [
  { id: 'row-1', role: 'user', parts: [textPart('List the repositories.')], rowId: 1 },
  { id: 'row-2', role: 'assistant', parts: [textPart('There are two repositories.')], rowId: 2 },
  { ...firstSend, rowId: undefined }
]

const lostRegenerate = {
  pending: [{ admission_id: 'adm-2', input_id: resentId, status: 'unknown', execution_generation: 2 }]
}

it('keeps the progress of a regenerate the owner lost in a crash, on every reopen', async () => {
  // The first send's bubble stays and the re-send runs as a new admission. The rewind cut the old turn's rows and
  // the restarted owner stored nothing of the re-sent one, so the journal alone holds the prompt and its progress.
  await streamThenDie(regenerated, 'adm-2', 'Reading the issue ')

  expect(await reopen(lostRegenerate, earlierExchange)).toEqual(['There are two repositories.', 'Reading the issue'])

  cleanup()
  clearAllSessionStates()

  expect(await reopen(lostRegenerate, earlierExchange)).toEqual(['There are two repositories.', 'Reading the issue'])
  expect(readInFlightTurnJournal(storedId)).not.toBeNull()
})

it('keeps a crash-lost regenerate recoverable after the reopened chat is activated again', async () => {
  // Switching away and back warm-activates the session; the owner still lists the lost turn and runs nothing.
  await streamThenDie(regenerated, 'adm-2', 'Reading the issue ')
  vi.mocked(getLatestSessionMessages).mockResolvedValue({ session_id: storedId, messages: earlierExchange })

  const live = mount({
    session_id: runtimeId,
    resumed: storedId,
    messages: [],
    messages_omitted: true,
    message_count: earlierExchange.length,
    running: false,
    ...lostRegenerate
  } as SessionResumeResult)

  for (const _open of ['cold resume', 'warm activation']) {
    await act(async () => {
      await live.result.current.actions.resumeSession(storedId, true)
    })
    await act(async () => {
      await vi.advanceTimersByTimeAsync(600)
    })
  }

  expect(live.requestGateway.mock.calls.map(([method]) => method)).toContain('session.activate')
  cleanup()
  clearAllSessionStates()

  expect(await reopen(lostRegenerate, earlierExchange)).toEqual(['There are two repositories.', 'Reading the issue'])
})

it('retires a finished queue drain while its successor waits instead of replaying it', async () => {
  // The drained prompt has no bubble, so the journal opens at the previous turn's stored prompt, which names no
  // admission. By reopen the drain has finished and that prompt is off the newest page.
  await streamThenDie(
    [
      { id: 'row-1', role: 'user', parts: [textPart('Summarize the open issues.')], rowId: 1 },
      { ...firstReply, id: 'row-4' }
    ],
    'adm-2',
    'Grouping '
  )

  const replies = await reopen({ pending: [{ admission_id: 'adm-3', input_id: resentId, status: 'queued' }] }, [
    nextPrompt,
    { id: 6, role: 'assistant', content: 'Grouped by area.', timestamp: 15 }
  ])

  expect(replies).toEqual(['Grouped by area.'])
  expect(readInFlightTurnJournal(storedId)).toBeNull()
})

it("keeps a finished turn's admission off the successor running beside it", async () => {
  // The dead tab's journal names its execution; the successor's own frames have not arrived when the journal is
  // rewritten, and then the gateway dies with the successor.
  persistInFlightTurnState({
    awaitingResponse: false,
    busy: true,
    messages: [
      { ...firstSend, rowId: undefined },
      { id: 'assistant-stream-1', role: 'assistant', parts: [textPart('Reading the issue ')], pending: true }
    ],
    storedSessionId: storedId,
    streamId: 'assistant-stream-1',
    turnAdmissionId: 'adm-1',
    turnStartedAt: 10_000
  })
  await vi.advanceTimersByTimeAsync(500)
  vi.mocked(getLatestSessionMessages)
    .mockResolvedValueOnce({ session_id: storedId, messages: turnSoFar })
    .mockRejectedValueOnce(new Error('REST unavailable'))

  const live = mount({
    session_id: runtimeId,
    resumed: storedId,
    messages: [],
    messages_omitted: true,
    message_count: settledTurn.length + 1,
    running: true,
    turn_started_at: 30,
    inflight: { user: 'Now group them.', assistant: 'Grouping ', streaming: true },
    pending: [{ admission_id: 'adm-2', input_id: resentId, status: 'started' }]
  } as SessionResumeResult)

  await act(async () => {
    await live.result.current.actions.resumeSession(storedId, true)
  })
  await act(async () => {
    await vi.advanceTimersByTimeAsync(600)
  })
  cleanup()
  clearAllSessionStates()

  const lost = { pending: [{ admission_id: 'adm-2', input_id: resentId, status: 'unknown', execution_generation: 3 }] }

  expect(await reopen(lost, settledTurn)).toContain('Grouping')
})

it('names the turn a warm activation adopts, not the one this tab last streamed', async () => {
  // This tab streamed and finished adm-1; another device then started adm-2. Switching back warm-activates the
  // session, and adm-2's progress so far arrives only in the activation snapshot. Then the gateway dies with adm-2.
  const anotherPrompt: SessionMessage = { id: 5, role: 'user', content: 'Draft the release notes.', timestamp: 30 }

  vi.mocked(getLatestSessionMessages).mockResolvedValue({
    session_id: storedId,
    messages: [...settledTurn, anotherPrompt]
  })

  const live = mount({
    session_id: runtimeId,
    resumed: storedId,
    messages: [],
    messages_omitted: true,
    message_count: settledTurn.length + 1,
    running: true,
    turn_started_at: 30,
    inflight: { user: 'Draft the release notes.', assistant: 'Only in the snapshot ', streaming: true },
    pending: [{ admission_id: 'adm-2', input_id: resentId, status: 'started' }]
  } as SessionResumeResult)

  act(() => {
    live.result.current.cache.updateSessionState(runtimeId, state => ({ ...state, messages: [firstSend] }), storedId)
  })

  for (const event of [
    { type: 'message.start', payload: {} },
    { type: 'message.delta', payload: { text: 'There are four open issues.' } },
    { type: 'message.complete', payload: { text: 'There are four open issues.' } }
  ]) {
    act(() =>
      live.result.current.stream.handleGatewayEvent({
        session_id: runtimeId,
        ...execution('adm-1'),
        ...event
      } as GatewayEvent)
    )
  }

  await act(async () => {
    await live.result.current.actions.resumeSession(storedId, true)
  })
  await act(async () => {
    await vi.advanceTimersByTimeAsync(600)
  })
  cleanup()
  clearAllSessionStates()

  const lost = { pending: [{ admission_id: 'adm-2', input_id: resentId, status: 'unknown', execution_generation: 3 }] }

  expect(await reopen(lost, [...settledTurn, anotherPrompt])).toContain('Only in the snapshot')
})
