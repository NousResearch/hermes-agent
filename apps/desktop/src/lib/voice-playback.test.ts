import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('@hermes/shared', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  // Return a URL that does not end in /api/ws so `resolveSpeakStreamUrl`
  // bails out before constructing any real WebSocket/AudioContext — this
  // test only needs to observe the `getConnection` call, not drive a full
  // streaming session.
  resolveGatewayWsUrl: vi.fn().mockResolvedValue('ws://gateway.example/other')
}))

import { $activeGatewayProfile } from '@/store/profile'

import { startSpeechStream } from './voice-playback'

describe('startSpeechStream connection scoping', () => {
  let getConnection: ReturnType<typeof vi.fn>

  beforeEach(() => {
    getConnection = vi.fn().mockResolvedValue({ authMode: 'token', baseUrl: 'http://gateway.example', token: 't' })
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { api: vi.fn(), getConnection }
    })
  })

  afterEach(() => {
    vi.restoreAllMocks()
    Reflect.deleteProperty(window, 'hermesDesktop')
    $activeGatewayProfile.set('default')
  })

  it('scopes the connection to the active gateway profile', async () => {
    $activeGatewayProfile.set('work')

    const session = await startSpeechStream({ source: 'read-aloud' })

    expect(getConnection).toHaveBeenCalledWith('work')
    expect(session).toBeNull()
  })

  it('scopes to the default profile when none is active', async () => {
    const session = await startSpeechStream({ source: 'read-aloud' })

    expect(getConnection).toHaveBeenCalledWith('default')
    expect(session).toBeNull()
  })
})
