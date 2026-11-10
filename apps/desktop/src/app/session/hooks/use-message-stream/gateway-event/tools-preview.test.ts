import { type GatewayEvent, JsonRpcGatewayClient } from '@hermes/shared'
import { afterEach, expect, it, vi } from 'vitest'

import { type GatewayEventPayload, upsertToolPart } from '@/lib/chat-messages'
import { createClientSessionState } from '@/lib/chat-runtime'
import { $previewStatusBySession, dismissPreviewArtifact, recordPreviewArtifact } from '@/store/preview-status'
import {
  $sessionStates,
  clearAllSessionStates,
  publishSessionState,
  recordSessionEventScope
} from '@/store/session-states'

import { handleToolEvent } from './tools'
import type { GatewayEventContext } from './types'

const sid = 'preview-production'
const args = { path: '/work/report.html' }

function pending() {
  const state = createClientSessionState('production-stored')
  state.busy = true
  state.messages = [
    {
      id: 'live-tool-message',
      role: 'assistant',
      parts: upsertToolPart(
        [],
        {
          name: 'write_file',
          tool_id: 'reused-tool-id',
          args
        },
        'running',
        1
      )
    }
  ]
  publishSessionState(sid, state)
}

function event(seq: number, result: unknown = { verified: true }): GatewayEvent<'tool.complete'> {
  return {
    type: 'tool.complete',
    session_id: sid,
    seq,
    payload: { name: 'write_file', tool_id: 'reused-tool-id', args, result }
  }
}

function deliver(event: GatewayEvent<'tool.complete'>) {
  return handleToolEvent({
    event,
    payload: event.payload,
    sessionId: sid,
    occurredAt: 2,
    isActiveEvent: false,
    deps: {
      flushQueuedDeltas: vi.fn(),
      updateSessionState: vi.fn(),
      sessionInterrupted: () => false,
      upsertToolCall: () => {
        const state = $sessionStates.get()[sid]

        if (state) {
          publishSessionState(sid, {
            ...state,
            messages: state.messages.map(message => ({
              ...message,
              parts: upsertToolPart(message.parts, event.payload as GatewayEventPayload, 'complete', 2)
            }))
          })
        }
      }
    }
  } as unknown as GatewayEventContext)
}

afterEach(() => {
  $previewStatusBySession.set({})
  clearAllSessionStates()
  window.localStorage.clear()
})

it('reoffers a real later production, never a historical mount or duplicate completion', () => {
  recordSessionEventScope({ session_id: sid, connectionId: 'local', profile: 'default' })
  pending()
  deliver(event(10))
  expect($previewStatusBySession.get()[sid]).toHaveLength(1)
  dismissPreviewArtifact(sid, '/work/report.html')
  recordPreviewArtifact(sid, 'file:///work/report.html', '/work')
  deliver(event(10))
  pending()
  deliver(event(11, { error: 'Permission denied' }))
  expect($previewStatusBySession.get()[sid]).toBeUndefined()
  pending()
  deliver(event(12))
  expect($previewStatusBySession.get()[sid]).toHaveLength(1)
  dismissPreviewArtifact(sid, '/work/report.html')
  deliver(event(12))
  expect($previewStatusBySession.get()[sid]).toBeUndefined()
})

class Socket extends EventTarget {
  readyState = 0
  sent: string[] = []
  send(data: string) {
    this.sent.push(data)
  }
  close() {
    this.readyState = 3
    this.dispatchEvent(new CloseEvent('close'))
  }
  open() {
    this.readyState = 1
    this.dispatchEvent(new Event('open'))
  }
  frame(data: unknown) {
    this.dispatchEvent(new MessageEvent('message', { data: JSON.stringify(data) }))
  }
}

it('keeps dismissal when a missed completion arrives through the real reconnect replay path', async () => {
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

  client.on('tool.complete', deliver)

  try {
    let connecting = client.connect('ws://fixture.invalid')
    sockets[0].open()
    await connecting
    sockets[0].frame({
      jsonrpc: '2.0',
      method: 'event',
      params: { type: 'message.delta', session_id: sid, seq: 1, payload: { text: '' } }
    })
    client.invalidate('fixture disconnect')
    connecting = client.connect('ws://fixture.invalid')
    sockets[1].open()
    await connecting
    await vi.waitFor(() =>
      expect(sockets[1].sent.map(text => JSON.parse(text).method)).toContain('session.events.since')
    )
    const request = sockets[1].sent.map(text => JSON.parse(text)).find(value => value.method === 'session.events.since')
    recordSessionEventScope({ session_id: sid, connectionId: 'local', profile: 'default' })
    pending()
    recordPreviewArtifact(sid, '/work/report.html', '/work', 'production-stored')
    dismissPreviewArtifact(sid, '/work/report.html', 'production-stored')
    const seen = vi.fn()
    client.on('tool.complete', seen)
    sockets[1].frame({
      jsonrpc: '2.0',
      id: request.id,
      result: {
        events: [event(2)],
        latest_seq: 2,
        truncated: false,
        count: 1
      }
    })
    await vi.waitFor(() => expect(seen).toHaveBeenCalledOnce())
    expect(seen.mock.calls[0][0].replayed).toBe(true)
    expect($previewStatusBySession.get()[sid]).toBeUndefined()
  } finally {
    client.close()
  }
})
