import { type GatewayEvent, JsonRpcGatewayClient } from '@hermes/shared'
import { act, cleanup, waitFor } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import type { ClientSessionState } from '@/app/types'
import { getLatestSessionMessages } from '@/hermes'
import { chatMessageText } from '@/lib/chat-messages'
import { createClientSessionState } from '@/lib/chat-runtime'
import { setPrimaryGateway } from '@/store/gateway'
import { setActiveSessionId, setBusy, setSelectedStoredSessionId } from '@/store/session'
import { clearAllSessionStates, publishSessionState } from '@/store/session-states'

import { renderMessageStream } from '../../session/hooks/use-message-stream/test-harness'

import {
  hydrateStoredSessionTranscript,
  reconcileActiveTranscript,
  reconcileTileTranscripts
} from './use-background-sync'

vi.mock('@/hermes', async original => ({
  ...(await original<Record<string, unknown>>()),
  getLatestSessionMessages: vi.fn()
}))

const sid = 'replay-hydration-runtime'
const stored = 'replay-hydration-stored'

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
  return { session_id: sid, seq, type, payload: { timestamp: seq, ...payload } } as GatewayEvent
}

async function setup(surface: 'active' | 'tile' | 'fallback') {
  setActiveSessionId(surface === 'tile' ? 'another-runtime' : sid)
  setSelectedStoredSessionId(stored)
  const states = new Map<string, ClientSessionState>()

  const updateSessionState = (id: string, updater: (state: ClientSessionState) => ClientSessionState) => {
    const next = updater(states.get(id) ?? createClientSessionState(stored))
    states.set(id, next)
    publishSessionState(id, next)

    return next
  }

  const stream = renderMessageStream(sid, { states, updateSessionState })
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

  setPrimaryGateway(client)
  client.onEvent(stream.handleEvent)

  const connect = async () => {
    const promise = client.connect('ws://fixture.test')
    const socket = sockets.at(-1)!
    socket.open()
    await promise

    return socket
  }

  const first = await connect()
  act(() => first.event(event(1, 'session.info', { running: false })))
  client.invalidate()
  const second = await connect()
  const request = second.sent.find(item => item.method === 'session.events.since')!
  expect(request.params.last_seen).toBe(1)
  const signatureRef = { current: new Map<string, string>() }
  const requestSequenceRef = { current: 0 }
  const activeSessionIdRef = { current: sid }
  const selectedStoredSessionIdRef = { current: stored }

  const refresh = () =>
    surface === 'tile'
      ? reconcileTileTranscripts({
          tiles: [{ storedSessionId: stored, runtimeId: sid }],
          signatureRef,
          requestSequenceRef,
          updateSessionState
        })
      : surface === 'fallback'
        ? hydrateStoredSessionTranscript({
            attempts: 1,
            storedSessionId: stored,
            runtimeSessionId: sid,
            storedProfile: 'default',
            updateSessionState
          })
        : reconcileActiveTranscript({
            activeSessionIdRef,
            selectedStoredSessionIdRef,
            busyRef: { current: false },
            requestSequenceRef,
            signatureRef,
            resolveSession: () => ({ profile: 'default' }),
            updateSessionState
          })

  return { client, stream, refresh, second, request, connect, activeSessionIdRef, selectedStoredSessionIdRef }
}

const replay = [
  event(2, 'message.start'),
  event(3, 'message.delta', { text: 'Earlier progress.' }),
  event(4, 'message.interim', { text: 'Earlier progress.', already_streamed: true }),
  event(5, 'tool.start', { name: 'read_file', tool_id: 'old-call', args: { path: 'example.py' } }),
  event(6, 'tool.complete', { name: 'read_file', tool_id: 'old-call', result: 'fixture contents' }),
  event(7, 'message.delta', { text: 'Finished result.' }),
  event(8, 'message.complete', { text: 'Finished result.' }),
  event(9, 'session.info', { running: false })
]

function stubHistory() {
  vi.mocked(getLatestSessionMessages).mockResolvedValue({
    session_id: stored,
    messages: [
      { id: 1, role: 'user', content: 'Review example', timestamp: 1 },
      {
        id: 2,
        role: 'assistant',
        content: 'Earlier progress.',
        timestamp: 2,
        tool_calls: [
          { id: 'old-call', type: 'function', function: { name: 'read_file', arguments: '{"path":"example.py"}' } }
        ]
      },
      {
        id: 3,
        role: 'tool',
        content: 'fixture contents',
        timestamp: 3,
        tool_call_id: 'old-call',
        tool_name: 'read_file'
      },
      { id: 4, role: 'assistant', content: 'Finished result.', timestamp: 4 }
    ]
  })
}

afterEach(() => {
  cleanup()
  clearAllSessionStates()
  setPrimaryGateway(null)
  setActiveSessionId(null)
  setSelectedStoredSessionId(null)
  setBusy(false)
  vi.mocked(getLatestSessionMessages).mockReset()
})

it.each(['active', 'tile', 'fallback'] as const)(
  'orders %s history after full replay without duplicate turns',
  async surface => {
    const { client, stream, refresh, second, request } = await setup(surface)

    try {
      stubHistory()
      let pending!: Promise<void>
      await act(async () => {
        pending = refresh()
      })
      // The database is already complete; publishing it ahead of the replay is
      // precisely the race. Do not issue the read until this session catches up.
      expect(getLatestSessionMessages).not.toHaveBeenCalled()
      await act(async () => {
        second.frame({ id: request.id, jsonrpc: '2.0', result: { events: replay } })
        await pending
      })
      const texts = stream.state().messages.map(chatMessageText).join('\n')

      expect(texts.match(/Earlier progress\./g)).toHaveLength(1)
      expect(texts.match(/Finished result\./g)).toHaveLength(1)
      expect(texts).toContain('Review example')
      expect(stream.state().busy).toBe(false)
    } finally {
      client.close()
    }
  }
)

it('keeps an unseen live turn parked behind replay instead of painting an idle snapshot over it', async () => {
  const { client, stream, refresh, second, request } = await setup('active')

  try {
    stubHistory()
    let pending!: Promise<void>
    await act(async () => {
      pending = refresh()
    })
    act(() => {
      second.event(event(10, 'message.start'))
      second.event(event(11, 'message.delta', { text: 'New work.' }))
    })
    await act(async () => {
      second.frame({ id: request.id, jsonrpc: '2.0', result: { events: replay } })
      await pending
    })
    expect(getLatestSessionMessages).not.toHaveBeenCalled()
    expect(stream.state().busy).toBe(true)
    // Replay dispatch does not flush the renderer's coalesced delta timer.
    await waitFor(() => expect(stream.state().messages.map(chatMessageText).join('\n')).toContain('New work.'))
  } finally {
    client.close()
  }
})

it('abandons a queued history read when its replay socket is invalidated', async () => {
  const { client, refresh } = await setup('active')
  stubHistory()
  let pending!: Promise<void>
  await act(async () => {
    pending = refresh()
  })
  await act(async () => {
    client.invalidate()
    await pending
  })
  expect(getLatestSessionMessages).not.toHaveBeenCalled()
  client.close()
})

it('does not follow a stale selection after waiting for replay', async () => {
  const { client, refresh, second, request, selectedStoredSessionIdRef } = await setup('active')

  try {
    stubHistory()
    let pending!: Promise<void>
    await act(async () => {
      pending = refresh()
    })
    selectedStoredSessionIdRef.current = 'another-session'
    await act(async () => {
      second.frame({ id: request.id, jsonrpc: '2.0', result: { events: replay } })
      await pending
    })
    expect(getLatestSessionMessages).not.toHaveBeenCalled()
  } finally {
    client.close()
  }
})

it('checks replay again when a socket reconnects during an existing history read', async () => {
  const { client, stream, refresh, second, request, connect } = await setup('active')

  try {
    await act(async () => second.frame({ id: request.id, jsonrpc: '2.0', result: { events: [] } }))
    stubHistory()
    const snapshot = await getLatestSessionMessages(stored)
    vi.mocked(getLatestSessionMessages).mockReset()
    let release!: () => void
    vi.mocked(getLatestSessionMessages).mockReturnValue(
      new Promise(resolve => {
        release = () => resolve(snapshot)
      })
    )
    let pending!: Promise<void>
    await act(async () => {
      pending = refresh()
    })
    expect(getLatestSessionMessages).toHaveBeenCalledTimes(1)
    client.invalidate()
    const next = await connect()
    const replayRequest = next.sent.find(item => item.method === 'session.events.since')!
    await act(async () => {
      release()
    })
    // The held REST result must not paint a completed turn ahead of replay.
    expect(stream.state().messages.map(chatMessageText).join('\n')).not.toContain('Finished result.')
    await act(async () => {
      next.frame({ id: replayRequest.id, jsonrpc: '2.0', result: { events: replay } })
      await pending
    })
    stubHistory()
    await act(refresh)
    const text = stream.state().messages.map(chatMessageText).join('\n')
    expect(text).toContain('Review example')
    expect(text.match(/Finished result\./g)).toHaveLength(1)
  } finally {
    client.close()
  }
})
