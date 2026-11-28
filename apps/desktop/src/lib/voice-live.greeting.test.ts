import { afterEach, describe, expect, it, vi } from 'vitest'

import { type VoiceLiveHandlers, VoiceLiveSession } from './voice-live'

// `voice.gpt_live.greeting` comes back in the session POST response and is said once,
// on the first `session.started`, as a commentary event. Empty means nothing is sent.

const handlers: VoiceLiveHandlers = {
  onClosed: () => undefined,
  onDelegation: () => undefined,
  onError: () => undefined
}

async function startWith(greeting: string | undefined) {
  const sent: Record<string, unknown>[] = []
  let deliver: (event: unknown) => void = () => undefined

  const channel = {
    addEventListener: (type: string, listener: (event: { data: string }) => void) => {
      if (type === 'message') {
        deliver = event => listener({ data: JSON.stringify(event) })
      }
    },
    readyState: 'open' as const,
    send: (raw: string) => sent.push(JSON.parse(raw))
  }

  const PeerConnection = class {
    addEventListener = () => undefined
    connectionState = 'new'
    createDataChannel = () => channel
    createOffer = async () => ({})
    iceGatheringState = 'complete'
    localDescription = { sdp: 'v=0\r\n' }
    setLocalDescription = async () => undefined
    setRemoteDescription = async () => undefined
  }

  Object.defineProperty(globalThis, 'RTCPeerConnection', { configurable: true, value: PeerConnection })
  Object.defineProperty(window.navigator, 'mediaDevices', {
    configurable: true,
    value: { getUserMedia: async () => ({ getAudioTracks: () => [] }) }
  })

  const api = vi.fn(async () => ({
    greeting,
    ok: true,
    session: { id: 's1' },
    transport: { sdp: 'answer', type: 'answer' }
  }))

  Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { api } })

  const session = new VoiceLiveSession(handlers)
  await session.start([])

  return { deliver: () => deliver({ session: { id: 's1' }, type: 'session.started' }), sent }
}

afterEach(() => {
  Reflect.deleteProperty(window, 'hermesDesktop')
  Reflect.deleteProperty(globalThis, 'RTCPeerConnection')
  Reflect.deleteProperty(window.navigator, 'mediaDevices')
})

describe('GPT-Live greeting', () => {
  it('says the greeting once when the call connects, even if session.started repeats', async () => {
    const { deliver, sent } = await startWith('  Hi, go ahead.  ')

    expect(sent).toEqual([])
    deliver()
    deliver()

    expect(sent).toHaveLength(1)
    expect(sent[0]).toMatchObject({ content: 'Hi, go ahead.', delegation_id: null, type: 'session.commentary.append' })
  })

  it.each([undefined, '', '   '])('sends nothing when the greeting is %j', async greeting => {
    const { deliver, sent } = await startWith(greeting)

    deliver()

    expect(sent).toEqual([])
  })
})
