import { afterEach, describe, expect, it, vi } from 'vitest'

import { openVoiceSocket } from './voice-stream'

// A local gateway admits its voice sockets on the /api/ws dial's grant, in the gateway
// subprotocol only (RT-71, RT-80); a remote route keeps its own query credential.
describe('openVoiceSocket', () => {
  const dials: unknown[][] = []

  afterEach(() => {
    dials.length = 0
    vi.unstubAllGlobals()
  })

  const open = (url: string) => {
    vi.stubGlobal('WebSocket', function (...args: unknown[]) {
      dials.push(args)
    })
    openVoiceSocket(url)

    return dials[0]
  }

  it('moves a local dial ticket out of the URL into the gateway subprotocol', () => {
    expect(open('ws://127.0.0.1:5151/api/audio/speak-stream?native_dial=n1&ticket=t1&profile=beta')).toEqual([
      'ws://127.0.0.1:5151/api/audio/speak-stream?native_dial=n1&profile=beta',
      ['hermes-gateway-v1', 'hermes-gateway-ticket.t1']
    ])
  })

  it('leaves a remote credential where the remote gateway reads it', () => {
    expect(open('wss://gateway.example/api/audio/transcribe-stream?ticket=fresh')).toEqual([
      'wss://gateway.example/api/audio/transcribe-stream?ticket=fresh'
    ])
  })
})
