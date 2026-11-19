import type { GatewayEvent } from '@hermes/shared'
import { QueryClient } from '@tanstack/react-query'
import { act, cleanup, renderHook } from '@testing-library/react'
import { useRef } from 'react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { getLatestSessionMessages } from '@/hermes'
import { type ChatMessage, chatMessageText, toChatMessages } from '@/lib/chat-messages'
import { persistInFlightTurnState, resetInFlightTurnJournalStateForTests } from '@/lib/inflight-turn-journal'
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

// A window reloads (or the app restarts) while a reply streams. The fresh
// renderer resumes the chat cold: the turn's live row comes back from the
// in-flight journal or the backend's in-flight projection, and the deltas the
// agent sends afterwards must extend that row, not open a second bubble beside it.
const storedId = 'reload-stream-target'
const runtimeId = 'reload-runtime'
const noop = async () => undefined

const storedRows: SessionMessage[] = [
  { id: 1, role: 'user', content: 'U1 first', timestamp: 10 },
  { id: 2, role: 'assistant', content: 'A1 one', timestamp: 11 },
  { id: 3, role: 'user', content: 'U2 second', timestamp: 20 }
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

    const stream = useMessageStream({
      ...cache,
      queryClient,
      hydrateFromStoredSession: noop,
      refreshHermesConfig: noop,
      refreshSessions: noop
    })

    return { cache, actions, stream }
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
  vi.mocked(getLatestSessionMessages).mockResolvedValue({ session_id: storedId, messages: storedRows })
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

/** The window before the reload journaled its live row (throttled write). */
async function journalLiveRowBeforeReload() {
  const live: ChatMessage[] = [
    ...toChatMessages(storedRows),
    { id: 'assistant-stream-before-reload', role: 'assistant', parts: [{ type: 'text', text: 'A2 ' }], pending: true }
  ]

  persistInFlightTurnState({
    awaitingResponse: false,
    busy: true,
    messages: live,
    storedSessionId: storedId,
    streamId: 'assistant-stream-before-reload',
    turnStartedAt: 19_000
  })
  await vi.advanceTimersByTimeAsync(500)
}

it.each([
  ['the in-flight journal', true, undefined],
  ['the backend in-flight projection', false, { user: 'U2 second', assistant: 'A2 ', streaming: true }]
] as const)(
  'a reply streaming across a reload extends the live row %s restored',
  async (_label, journaled, inflight) => {
    if (journaled) {
      await journalLiveRowBeforeReload()
    }

    const { result } = mount({
      session_id: runtimeId,
      resumed: storedId,
      messages: storedRows,
      message_count: 3,
      running: true,
      turn_started_at: 19,
      ...(inflight ? { inflight } : {})
    } as SessionResumeResult)

    await act(async () => {
      await result.current.actions.resumeSession(storedId, true)
    })

    const send = (type: GatewayEvent['type'], payload: GatewayEvent['payload']) =>
      act(() => result.current.stream.handleGatewayEvent({ session_id: runtimeId, type, payload } as GatewayEvent))

    send('message.delta', { text: 'finished after the reload' })
    await act(async () => {
      await vi.advanceTimersByTimeAsync(150)
    })

    const replies = () => {
      const rows = result.current.cache.sessionStateByRuntimeIdRef.current.get(runtimeId)!.messages
      const prompt = rows.findLastIndex(row => row.role === 'user')

      return rows.slice(prompt + 1).filter(row => row.role === 'assistant' && chatMessageText(row).trim())
    }

    expect(replies().map(row => chatMessageText(row).trim())).toEqual(['A2 finished after the reload'])

    send('message.complete', { text: 'A2 finished after the reload', status: 'complete' })
    await act(async () => {
      await vi.advanceTimersByTimeAsync(150)
    })

    expect(replies().map(row => chatMessageText(row).trim())).toEqual(['A2 finished after the reload'])
  }
)
