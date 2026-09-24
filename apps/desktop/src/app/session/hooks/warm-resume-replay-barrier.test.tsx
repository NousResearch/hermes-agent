import { type GatewayEvent, JsonRpcGatewayClient } from '@hermes/shared'
import { QueryClient } from '@tanstack/react-query'
import { act, cleanup, renderHook } from '@testing-library/react'
import { useRef } from 'react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import type { HermesGateway } from '@/api/client'
import { reconcileActiveTranscript } from '@/app/contrib/hooks/use-background-sync'
import { getLatestSessionMessages } from '@/hermes'
import { chatMessageText, toChatMessages } from '@/lib/chat-messages'
import { resetInFlightTurnJournalStateForTests } from '@/lib/inflight-turn-journal'
import { setPrimaryGateway } from '@/store/gateway'
import { $activeGatewayProfile } from '@/store/profile'
import {
  $messages,
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

const storedId = 'warm-replay-stored'
const runtimeId = 'warm-replay-runtime'
const noop = async () => undefined
const user: SessionMessage = { id: 1, role: 'user', content: 'Review example', timestamp: 1 }

class Socket extends EventTarget {
  readyState = 0
  sent: { id: string; method: string; params: Record<string, unknown> }[] = []
  send(data: string) {
    this.sent.push(JSON.parse(data))
  }
  close() {
    this.readyState = 3
    this.dispatchEvent(new CloseEvent('close'))
  }
  open() {
    this.readyState = 1
    this.dispatchEvent(new Event('open'))
  }
  frame(frame: unknown) {
    this.dispatchEvent(new MessageEvent('message', { data: JSON.stringify(frame) }))
  }
  event(event: GatewayEvent) {
    this.frame({ jsonrpc: '2.0', method: 'event', params: event })
  }
}

function event(seq: number, type: GatewayEvent['type'], payload: Record<string, unknown> = {}): GatewayEvent {
  return { session_id: runtimeId, seq, type, payload: { timestamp: seq, ...payload } } as GatewayEvent
}

const replay = [
  event(2, 'message.start'),
  event(3, 'message.delta', { text: 'Finished result.' }),
  event(4, 'message.complete', { text: 'Finished result.' }),
  event(5, 'session.info', { running: false })
]

const snapshot: SessionResumeResult = {
  session_id: runtimeId,
  resumed: storedId,
  messages: [],
  message_count: 0,
  running: false
}

async function mountWithPendingReplay() {
  const requestGateway = vi.fn().mockResolvedValue(snapshot)

  const hook = renderHook(() => {
    const busyRef = useRef(false)
    const creatingSessionRef = useRef(false)
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
      creatingSessionRef,
      getRouteToken: () => 'A',
      getRoutedStoredSessionId: () => null,
      navigate: vi.fn(),
      requestGateway
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

  const sockets: Socket[] = []

  const client = new JsonRpcGatewayClient({
    heartbeatIntervalMs: 0,
    heartbeatDeadlineMs: 0,
    socketFactory: () => {
      const socket = new Socket()
      sockets.push(socket)

      return socket as unknown as WebSocket
    }
  })

  // The store types the primary as the desktop subclass; the barrier under test lives on the shared base.
  setPrimaryGateway(client as unknown as HermesGateway)
  client.onEvent(gatewayEvent => hook.result.current.stream.handleGatewayEvent(gatewayEvent))

  const connect = async () => {
    const promise = client.connect('ws://fixture.test')
    const socket = sockets.at(-1)!
    socket.open()
    await promise

    return socket
  }

  // Warm cache: the runtime is already known and has seen seq 1.
  act(() => {
    hook.result.current.cache.updateSessionState(
      runtimeId,
      state => ({ ...state, messages: toChatMessages([user]) }),
      storedId
    )
  })
  const first = await connect()
  act(() => first.event(event(1, 'session.info', { running: false })))
  client.invalidate()
  const second = await connect()
  const request = second.sent.find(item => item.method === 'session.events.since')!
  expect(request.params).toMatchObject({ session_id: runtimeId, last_seen: 1 })

  return { ...hook, client, connect, requestGateway, second, request }
}

beforeEach(() => {
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
  vi.mocked(getLatestSessionMessages).mockResolvedValue({
    session_id: storedId,
    messages: [user, { id: 2, role: 'assistant', content: 'Finished result.', timestamp: 2 }]
  })
})

afterEach(() => {
  cleanup()
  clearAllSessionStates()
  resetInFlightTurnJournalStateForTests()
  setPrimaryGateway(null)
  localStorage.clear()
  setSessions([])
  setMessages([])
  setActiveSessionId(null)
  setSelectedStoredSessionId(null)
  setBusy(false)
  setAwaitingResponse(false)
  vi.restoreAllMocks()
})

const activateCalls = (requestGateway: ReturnType<typeof vi.fn>) =>
  requestGateway.mock.calls.filter(([method]) => method === 'session.activate')

it('waits for reconnect replay before activating and reading warm history', async () => {
  const { result, client, requestGateway, second, request } = await mountWithPendingReplay()

  try {
    let pending!: Promise<void>
    await act(async () => {
      pending = result.current.actions.resumeSession(storedId, true)
    })
    // The completed turn is already durable; neither the activation snapshot
    // nor REST history may publish it ahead of the replay that carries it.
    expect(activateCalls(requestGateway)).toHaveLength(0)
    expect(getLatestSessionMessages).not.toHaveBeenCalled()

    await act(async () => {
      second.frame({ id: request.id, jsonrpc: '2.0', result: { events: replay } })
      await pending
    })

    expect(activateCalls(requestGateway)).toHaveLength(1)
    const rows = result.current.cache.sessionStateByRuntimeIdRef.current.get(runtimeId)!.messages
    const text = rows.map(chatMessageText).join('\n')
    expect(text).toContain('Review example')
    expect(text.match(/Finished result\./g)).toHaveLength(1)
  } finally {
    client.close()
  }
})

it('still settles a warm resume through session.activate when its replay socket is lost', async () => {
  const { result, client, requestGateway } = await mountWithPendingReplay()

  requestGateway.mockImplementation(async (method: string) => {
    if (method === 'session.activate') {
      throw new Error('session not found')
    }

    return snapshot
  })

  try {
    let pending!: Promise<void>
    await act(async () => {
      pending = result.current.actions.resumeSession(storedId, true)
    })
    expect(activateCalls(requestGateway)).toHaveLength(0)

    await act(async () => {
      client.invalidate()
      await pending
    })

    // A lost socket is not proof the runtime is alive or gone: activation
    // decides, and a gone runtime falls back to a cold resume instead of
    // leaving the view on an unrebound warm cache.
    expect(activateCalls(requestGateway)).toHaveLength(1)
    expect(requestGateway.mock.calls.some(([method]) => method === 'session.resume')).toBe(true)
  } finally {
    client.close()
  }
})

it('holds a reconnect re-resume of the selected session behind its replay (cold path)', async () => {
  const { result, client, requestGateway, second, request } = await mountWithPendingReplay()
  const { activeSessionIdRef, runtimeIdByStoredSessionIdRef, selectedStoredSessionIdRef } = result.current.cache

  // The view is streaming this runtime, but the warm mapping is gone, so the
  // route's reconnect re-resume takes the cold REST + session.resume path.
  selectedStoredSessionIdRef.current = storedId
  activeSessionIdRef.current = runtimeId
  runtimeIdByStoredSessionIdRef.current.delete(storedId)

  try {
    let pending!: Promise<void>
    await act(async () => {
      pending = result.current.actions.resumeSession(storedId, true)
    })
    expect(getLatestSessionMessages).not.toHaveBeenCalled()
    expect(requestGateway).not.toHaveBeenCalled()

    await act(async () => {
      second.frame({ id: request.id, jsonrpc: '2.0', result: { events: replay } })
      await pending
    })

    expect(getLatestSessionMessages).toHaveBeenCalledTimes(1)
    expect(requestGateway.mock.calls.some(([method]) => method === 'session.resume')).toBe(true)
    expect(
      $messages
        .get()
        .map(chatMessageText)
        .join('\n')
        .match(/Finished result\./g)
    ).toHaveLength(1)
  } finally {
    client.close()
  }
})

it('does not paint a cold re-resume read ahead of a replay redialed while REST was in flight', async () => {
  const { result, client, connect } = await mountWithPendingReplay()
  const { activeSessionIdRef, runtimeIdByStoredSessionIdRef, selectedStoredSessionIdRef } = result.current.cache
  let releaseRest!: () => void
  const restGate = new Promise<void>(resolve => (releaseRest = resolve))
  const latest = vi.mocked(getLatestSessionMessages).getMockImplementation()!
  vi.mocked(getLatestSessionMessages).mockImplementation(async (...args) => restGate.then(() => latest(...args)))
  selectedStoredSessionIdRef.current = storedId
  activeSessionIdRef.current = runtimeId
  runtimeIdByStoredSessionIdRef.current.delete(storedId)

  try {
    let pending!: Promise<void>
    await act(async () => {
      pending = result.current.actions.resumeSession(storedId, true)
    })
    // The replay socket drops (barrier false: cold resume starts its REST read)
    // and redials before that read returns, owning a fresh replay of the turn.
    let third!: Awaited<ReturnType<typeof connect>>
    await act(async () => {
      client.invalidate()
      third = await connect()
    })
    expect(getLatestSessionMessages).toHaveBeenCalledTimes(1)
    const request = third.sent.find(item => item.method === 'session.events.since')!

    await act(async () => {
      releaseRest()
      await new Promise(resolve => setTimeout(resolve, 0))
      third.frame({ id: request.id, jsonrpc: '2.0', result: { events: replay } })
      await pending
    })

    expect(
      $messages
        .get()
        .map(chatMessageText)
        .join('\n')
        .match(/Finished result\./g)
    ).toHaveLength(1)
  } finally {
    client.close()
  }
})

it('holds a background active-transcript refresh behind the reconnect replay', async () => {
  const { result, client, second, request } = await mountWithPendingReplay()

  const { activeSessionIdRef, selectedStoredSessionIdRef, sessionStateByRuntimeIdRef, updateSessionState } =
    result.current.cache

  selectedStoredSessionIdRef.current = storedId
  activeSessionIdRef.current = runtimeId

  try {
    const refresh = reconcileActiveTranscript({
      activeSessionIdRef,
      busyRef: { current: false },
      requestSequenceRef: { current: 0 },
      resolveSession: () => ({ profile: 'default' }),
      selectedStoredSessionIdRef,
      signatureRef: { current: new Map() },
      updateSessionState
    })

    await new Promise(resolve => setTimeout(resolve, 0))
    expect(getLatestSessionMessages).not.toHaveBeenCalled()

    await act(async () => {
      second.frame({ id: request.id, jsonrpc: '2.0', result: { events: replay } })
      await refresh
    })

    expect(getLatestSessionMessages).toHaveBeenCalledTimes(1)
    const text = sessionStateByRuntimeIdRef.current.get(runtimeId)!.messages.map(chatMessageText).join('\n')
    expect(text.match(/Finished result\./g)).toHaveLength(1)
  } finally {
    client.close()
  }
})
