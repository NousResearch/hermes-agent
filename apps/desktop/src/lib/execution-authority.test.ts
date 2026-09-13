import { JsonRpcGatewayClient } from '@hermes/shared'
import type { GatewayEvent } from '@hermes/shared'
import { describe, expect, it } from 'vitest'

import { acceptExecutionEvent } from './execution-authority'
import frames from './execution-authority.frames.json'

/**
 * `execution-authority.frames.json` is captured from the real owner
 * (`tests/gateway/fixtures/execution_frames_capture.py`): two successive
 * authority epochs each running one turn on the same session, then an idle
 * `rename` mutation. Nothing here fabricates a wire shape.
 */
interface ServerFrame {
  params: GatewayEvent & { seq?: number }
}


const owner1 = (frames as { epoch1: { frames: ServerFrame[] } }).epoch1.frames
const owner2 = (frames as { epoch2: { frames: ServerFrame[]; idle_mutation_frames: ServerFrame[] } }).epoch2

class FakeWebSocket extends EventTarget {
  static OPEN = 1
  readyState = 1
  send(): void {}
  close(): void {}
}

function connectedClient(): { client: JsonRpcGatewayClient; socket: FakeWebSocket; received: GatewayEvent[] } {
  const socket = new FakeWebSocket()

  const client = new JsonRpcGatewayClient({
    socketFactory: () => socket as unknown as WebSocket,
    heartbeatIntervalMs: 0,
    heartbeatDeadlineMs: 0,
    connectTimeoutMs: 1000
  })

  const connecting = client.connect('ws://owner')
  socket.dispatchEvent(new Event('open'))
  void connecting
  const received: GatewayEvent[] = []
  client.onEvent(event => received.push(event))

  return { client, socket, received }
}

function deliver(socket: FakeWebSocket, frame: ServerFrame): void {
  socket.dispatchEvent(new MessageEvent('message', { data: JSON.stringify({ jsonrpc: '2.0', method: 'event', ...frame }) }))
}

function fence(authorities: Map<string, unknown>, event: GatewayEvent): boolean {
  return acceptExecutionEvent(authorities as never, 'session', event.type, event as unknown as Record<string, unknown>)
}

describe('execution authority fence over real owner frames', () => {
  it('installs authority from the first claimed frame and fences a retired epoch', () => {
    const { socket, received } = connectedClient()
    const authorities = new Map()
    const verdicts: Array<[string, boolean]> = []

    const stop = owner1.findIndex(frame => frame.params.type === 'message.complete')
    const retiredCompletion = owner1[stop]
    const owner1Live = owner1.slice(0, stop)
    const owner2Start = owner2.frames.findIndex(frame => frame.params.type === 'message.start')

    // Owner 1 starts a turn; owner 2 (a restarted gateway) takes the session over
    // before owner 1's completion reaches the viewer.
    for (const frame of [...owner1Live, ...owner2.frames.slice(owner2Start)]) { deliver(socket, frame) }
    deliver(socket, retiredCompletion)

    for (const event of received) { verdicts.push([event.type, fence(authorities, event)]) }

    const installed = authorities.get('session') as { epoch: string; generation: number; terminal: boolean; retiredEpochs: Set<string> }
    expect(installed).toBeDefined()
    expect(installed.epoch).toBe(String(owner2.frames[owner2Start].params.authority_epoch))
    expect(installed.generation).toBe(owner2.frames[owner2Start].params.execution_generation)
    expect(installed.retiredEpochs.has(String(owner1[1].params.authority_epoch))).toBe(true)
    // Every frame stamped by the live owners is admitted in order...
    expect(verdicts.slice(0, -1).every(([, ok]) => ok)).toBe(true)
    // ...and the retired owner's late completion is rejected and mutates nothing.
    expect(verdicts.at(-1)).toEqual(['message.complete', false])
    expect(authorities.get('session')).toBe(installed)
    expect(installed.terminal).toBe(true)
  })

  it('admits an idle session.updated after the terminal frame of the same owner', () => {
    const { socket, received } = connectedClient()
    const authorities = new Map()

    for (const frame of [...owner2.frames, ...owner2.idle_mutation_frames]) { deliver(socket, frame) }

    const verdicts = received.map(event => [event.type, fence(authorities, event)])
    expect(verdicts.at(-1)).toEqual(['session.updated', true])
    expect(verdicts.filter(([type]) => type === 'message.complete')).toEqual([['message.complete', true]])
  })
})
