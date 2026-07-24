import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

type ChannelListener = (event: MessageEvent) => void

class FakeBroadcastChannel {
  static instances = new Map<string, Set<FakeBroadcastChannel>>()

  name: string
  private listeners = new Set<ChannelListener>()

  constructor(name: string) {
    this.name = name
    const group = FakeBroadcastChannel.instances.get(name) ?? new Set()
    group.add(this)
    FakeBroadcastChannel.instances.set(name, group)
  }

  postMessage(data: unknown) {
    const group = FakeBroadcastChannel.instances.get(this.name)

    if (!group) {
      return
    }

    for (const channel of group) {
      if (channel === this) {
        continue
      }

      for (const listener of channel.listeners) {
        listener({ data } as MessageEvent)
      }
    }
  }

  addEventListener(_type: string, listener: ChannelListener) {
    this.listeners.add(listener)
  }

  removeEventListener(_type: string, listener: ChannelListener) {
    this.listeners.delete(listener)
  }
}

describe('transcript-sync', () => {
  beforeEach(() => {
    FakeBroadcastChannel.instances.clear()
    vi.stubGlobal('BroadcastChannel', FakeBroadcastChannel)
  })

  afterEach(() => {
    vi.unstubAllGlobals()
    vi.resetModules()
  })

  it('broadcastTranscriptChanged reaches peer windows on hermes:transcript', async () => {
    const { broadcastTranscriptChanged } = await import('./transcript-sync')

    const peer = new FakeBroadcastChannel('hermes:transcript')
    const received: unknown[] = []
    peer.addEventListener('message', event => received.push(event.data))

    broadcastTranscriptChanged({ sessionId: 'stored-1', messageCount: 4, updatedAt: 123 })

    expect(received).toEqual([{ sessionId: 'stored-1', messageCount: 4, updatedAt: 123 }])
  })

  it('onTranscriptChanged ignores malformed payloads', async () => {
    const { onTranscriptChanged } = await import('./transcript-sync')

    const received: unknown[] = []
    onTranscriptChanged(payload => received.push(payload))

    const peer = new FakeBroadcastChannel('hermes:transcript')
    peer.postMessage({ sessionId: 'x' })
    peer.postMessage(null)
    peer.postMessage({ sessionId: 'x', messageCount: 'nope' })
    peer.postMessage({ sessionId: 'stored-1', messageCount: 2 })

    expect(received).toEqual([{ sessionId: 'stored-1', messageCount: 2 }])
  })
})
