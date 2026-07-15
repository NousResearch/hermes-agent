import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

class FakeBroadcastChannel extends EventTarget {
  static channels: FakeBroadcastChannel[] = []

  constructor(readonly name: string) {
    super()
    FakeBroadcastChannel.channels.push(this)
  }

  close(): void {
    FakeBroadcastChannel.channels = FakeBroadcastChannel.channels.filter(channel => channel !== this)
  }

  postMessage(data: unknown): void {
    for (const channel of FakeBroadcastChannel.channels) {
      if (channel !== this && channel.name === this.name) {
        channel.dispatchEvent(new MessageEvent('message', { data }))
      }
    }
  }
}

describe('cross-window unread session sync', () => {
  beforeEach(() => {
    FakeBroadcastChannel.channels = []
    vi.stubGlobal('BroadcastChannel', FakeBroadcastChannel)
    vi.resetModules()
  })

  afterEach(() => {
    vi.unstubAllGlobals()
    vi.resetModules()
  })

  it('keeps legacy and typed session-list refresh messages compatible', async () => {
    const sync = await import('./session-sync')
    const remote = new FakeBroadcastChannel('hermes:sessions')
    const changed = vi.fn()
    const unsubscribe = sync.onSessionsChanged(changed)

    remote.postMessage(1)
    remote.postMessage({ type: 'sessions-changed' })
    remote.postMessage({ sessionId: 'stored-session', type: 'session-unread-changed', unread: true })

    expect(changed).toHaveBeenCalledTimes(2)
    unsubscribe()
  })

  it('mirrors unread updates between isolated renderer stores', async () => {
    const first = await import('./session')
    vi.resetModules()
    const second = await import('./session')

    first.setSessionUnread('stored-session', true)
    expect(second.$unreadFinishedSessionIds.get()).toEqual(['stored-session'])

    second.setSessionUnread('stored-session', false)
    expect(first.$unreadFinishedSessionIds.get()).toEqual([])
  })
})
