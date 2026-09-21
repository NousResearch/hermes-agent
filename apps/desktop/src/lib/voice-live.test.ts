import { afterEach, expect, it, vi } from 'vitest'

import { VoiceLiveSession } from './voice-live'
vi.mock('@/api/client', () => ({ profileScoped: () => ({}), ownerScoped: () => ({}) }))
vi.mock('@/hermes', () => ({ hermesApi: async () => ({ ok: true, transport: { sdp: 'answer' } }) }))
afterEach(() => {
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
  vi.useRealTimers()
})

function fixture(media: Promise<unknown>) {
  const channel = Object.assign(new EventTarget(), { readyState: 'open', send: vi.fn(), close: vi.fn() })

  const peer = Object.assign(new EventTarget(), {
    iceGatheringState: 'complete',
    localDescription: { sdp: 'offer' },
    addTrack: vi.fn(),
    createDataChannel: () => channel,
    createOffer: async () => ({}),
    setLocalDescription: async () => {},
    setRemoteDescription: async () => {},
    close: vi.fn()
  })

  vi.stubGlobal('RTCPeerConnection', function () {
    return peer
  })
  vi.stubGlobal('navigator', { mediaDevices: { getUserMedia: () => media } })
  vi.spyOn(HTMLMediaElement.prototype, 'pause').mockImplementation(() => {})
  const onClosed = vi.fn()
  const session = new VoiceLiveSession({ onClosed, onDelegation: vi.fn(), onError: vi.fn() })

  return { session, peer, channel, onClosed }
}

it('stops capture synchronously while retaining the close acknowledgement and usage', async () => {
  vi.useFakeTimers()
  const track = { stop: vi.fn() }

  const { session, channel, peer, onClosed } = fixture(
    Promise.resolve({ getAudioTracks: () => [track], getTracks: () => [track] })
  )

  await session.start([])
  session.close()
  expect(track.stop).toHaveBeenCalledOnce()
  expect(peer.close).not.toHaveBeenCalled()
  expect(onClosed).not.toHaveBeenCalled()
  expect(channel.send).toHaveBeenCalledWith(JSON.stringify({ type: 'session.close' }))
  channel.dispatchEvent(
    new MessageEvent('message', {
      data: JSON.stringify({ type: 'session.closed', reason: 'closed', usage: { seconds: 42 } })
    })
  )
  expect(onClosed).toHaveBeenCalledWith('closed', 42)
  expect(peer.close).toHaveBeenCalledOnce()
  await vi.runAllTimersAsync()
  expect(onClosed).toHaveBeenCalledOnce()
})
it('stops a microphone granted after close without attaching it to the peer', async () => {
  const track = { stop: vi.fn() }
  let grant!: (stream: unknown) => void

  const { session, peer } = fixture(
    new Promise(resolve => {
      grant = resolve
    })
  )

  const starting = session.start([])
  session.close()
  grant({ getAudioTracks: () => [track], getTracks: () => [track] })
  await starting
  expect(track.stop).toHaveBeenCalledOnce()
  expect(peer.addTrack).not.toHaveBeenCalled()
})
