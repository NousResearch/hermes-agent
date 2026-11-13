import { afterEach, describe, expect, it, vi } from 'vitest'

import { openGatewaySocket } from './canonical-protocol'

// Chat and the voice sockets beside it (RT-71, RT-80) share one dial rule: a local gateway's
// ticket rides only in the gateway subprotocol; a remote route keeps its own query credential.
describe('openGatewaySocket', () => {
  const dials: unknown[][] = []

  afterEach(() => {
    dials.length = 0
    vi.unstubAllGlobals()
  })

  const open = (url: string) => {
    vi.stubGlobal('WebSocket', function (...args: unknown[]) {
      dials.push(args)
    })
    openGatewaySocket(url)

    return dials[0]
  }

  it('moves a local dial ticket out of the URL into the gateway subprotocol', () => {
    expect(open('ws://127.0.0.1:5151/api/audio/speak-stream?native_dial=n1&ticket=t1&profile=beta')).toEqual([
      'ws://127.0.0.1:5151/api/audio/speak-stream?native_dial=n1&profile=beta',
      ['hermes-gateway-v1', 'hermes-gateway-ticket.t1']
    ])
  })

  it('refuses a local dial without a ticket', () => {
    expect(() => open('ws://127.0.0.1:5151/api/audio/transcribe-stream?native_dial=n1')).toThrow(/fresh private ticket/)
    expect(dials).toEqual([])
  })

  it('leaves a remote credential where the remote gateway reads it', () => {
    expect(open('wss://gateway.example/api/audio/transcribe-stream?ticket=fresh')).toEqual([
      'wss://gateway.example/api/audio/transcribe-stream?ticket=fresh'
    ])
  })
})
