import { JsonRpcGatewayClient } from '@hermes/shared'
import { act, cleanup } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { hydrateStoredSessionTranscript } from '@/app/contrib/hooks/use-background-sync'
import type { ClientSessionState } from '@/app/types'
import { getLatestSessionMessages } from '@/hermes'
import { chatMessageText } from '@/lib/chat-messages'
import { createClientSessionState } from '@/lib/chat-runtime'

import { renderMessageStream } from './test-harness'

// A reconnect replay the backend reports `truncated` (its ring evicted part of the gap) carries
// only the newest events. The client used to apply it as if complete, leaving a hole in the
// transcript; the session must be re-read from stored history instead.

vi.mock('@/hermes', async original => ({
  ...(await original<Record<string, unknown>>()),
  getLatestSessionMessages: vi.fn()
}))

const runtimeId = 'replay-runtime'
const storedId = 'replay-stored'

class Socket extends EventTarget {
  readyState = 0
  sent: { id: string; method: string }[] = []
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
}

afterEach(() => {
  cleanup()
  vi.mocked(getLatestSessionMessages).mockReset()
})

/** Connect a real client into the stream, see seq 3, drop, reconnect, and answer the replay. */
async function reconnectWithReplay(handleEvent: Parameters<JsonRpcGatewayClient['onEvent']>[0], truncated: boolean) {
  const sockets: Socket[] = []

  const client = new JsonRpcGatewayClient({
    heartbeatDeadlineMs: 0,
    heartbeatIntervalMs: 0,
    socketFactory: () => {
      const socket = new Socket()
      sockets.push(socket)

      return socket as unknown as WebSocket
    }
  })

  client.onEvent(event => act(() => handleEvent(event)))

  const first = client.connect('ws://gateway')
  sockets[0].open()
  await first
  sockets[0].frame({
    jsonrpc: '2.0',
    method: 'event',
    params: { payload: { running: false }, seq: 3, session_id: runtimeId, type: 'session.info' }
  })

  client.invalidate('drop')
  const second = client.connect('ws://gateway')
  sockets[1].open()
  await second

  await vi.waitFor(() => expect(sockets[1].sent.at(-1)?.method).toBe('session.events.since'))
  sockets[1].frame({
    jsonrpc: '2.0',
    id: sockets[1].sent.at(-1)?.id,
    result: {
      count: 1,
      events: [{ payload: { running: false }, seq: 900, session_id: runtimeId, type: 'session.info' }],
      latest_seq: 900,
      truncated
    }
  })

  await vi.waitFor(() => expect(client.sessionReplayBarrier(runtimeId)).toBeUndefined())

  return client
}

it.each([true, false])('re-reads stored history only after a truncated reconnect replay (truncated=%s)', async truncated => {
  const hydrateFromStoredSession = vi.fn(async () => undefined)
  const states = new Map([[runtimeId, { ...createClientSessionState(), storedSessionId: storedId }]])
  const stream = renderMessageStream(runtimeId, { hydrateFromStoredSession, states })

  const client = await reconnectWithReplay(stream.handleEvent, truncated)

  if (truncated) {
    expect(hydrateFromStoredSession).toHaveBeenCalledWith(expect.any(Number), storedId, runtimeId)
  } else {
    expect(hydrateFromStoredSession).not.toHaveBeenCalled()
  }

  client.close()
})

it('repairs the hole even when the first history read races the reconnect', async () => {
  // Nothing re-announces the hole once the watermark moved past it, so a read that fails while the
  // backend is still coming back must not be the only attempt.
  vi.mocked(getLatestSessionMessages)
    .mockRejectedValueOnce(new Error('backend restarting'))
    .mockResolvedValueOnce({
      messages: [
        { content: 'What happened while I was away?', id: 1, role: 'user', timestamp: 1 },
        { content: 'The evicted answer.', id: 2, role: 'assistant', timestamp: 2 }
      ],
      session_id: storedId
    })

  const states = new Map<string, ClientSessionState>([
    [runtimeId, { ...createClientSessionState(), storedSessionId: storedId }]
  ])

  const updateSessionState = (id: string, updater: (state: ClientSessionState) => ClientSessionState) => {
    const next = updater(states.get(id) ?? createClientSessionState())
    states.set(id, next)

    return next
  }

  const stream = renderMessageStream(runtimeId, {
    hydrateFromStoredSession: async (attempts = 1, stored = storedId, runtime = runtimeId) =>
      hydrateStoredSessionTranscript({
        attempts,
        runtimeSessionId: runtime ?? runtimeId,
        storedProfile: 'default',
        storedSessionId: stored ?? storedId,
        updateSessionState
      }),
    states,
    updateSessionState
  })

  const client = await reconnectWithReplay(stream.handleEvent, true)

  await vi.waitFor(() => {
    const messages = states.get(runtimeId)?.messages ?? []
    expect(messages.map(chatMessageText)).toContain('The evicted answer.')
  })
  expect(getLatestSessionMessages).toHaveBeenCalledTimes(2)

  client.close()
})
