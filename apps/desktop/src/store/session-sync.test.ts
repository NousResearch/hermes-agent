import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

class FakeBroadcastChannel extends EventTarget {
  static channels: FakeBroadcastChannel[] = []
  static heldMessages: { data: unknown; sender: FakeBroadcastChannel }[] = []
  static holdMessages = false

  constructor(readonly name: string) {
    super()
    FakeBroadcastChannel.channels.push(this)
  }

  close(): void {
    FakeBroadcastChannel.channels = FakeBroadcastChannel.channels.filter(channel => channel !== this)
  }

  postMessage(data: unknown): void {
    if (FakeBroadcastChannel.holdMessages) {
      FakeBroadcastChannel.heldMessages.push({ data, sender: this })

      return
    }

    FakeBroadcastChannel.deliver(this, data)
  }

  static deliver(sender: FakeBroadcastChannel, data: unknown): void {
    for (const channel of FakeBroadcastChannel.channels) {
      if (channel !== sender && channel.name === sender.name) {
        channel.dispatchEvent(new MessageEvent('message', { data }))
      }
    }
  }

  static flushReversed(): void {
    const messages = FakeBroadcastChannel.heldMessages.splice(0).reverse()
    FakeBroadcastChannel.holdMessages = false
    messages.forEach(({ data, sender }) => FakeBroadcastChannel.deliver(sender, data))
  }
}

describe('cross-window unread session sync', () => {
  beforeEach(() => {
    FakeBroadcastChannel.channels = []
    FakeBroadcastChannel.heldMessages = []
    FakeBroadcastChannel.holdMessages = false
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

  it('hydrates a late renderer from a snapshot and mirrors a gateway reset', async () => {
    const first = await import('./session')
    first.setSessionUnread('stored-session', true)

    vi.resetModules()
    const second = await import('./session')

    expect(second.$unreadFinishedSessionIds.get()).toEqual(['stored-session'])

    second.clearAllSessionUnread()

    expect(first.$unreadFinishedSessionIds.get()).toEqual([])
    expect(second.$unreadFinishedSessionIds.get()).toEqual([])
  })

  it('keeps a later no-op acknowledgement when an older completion arrives late', async () => {
    const first = await import('./session')
    vi.resetModules()
    const second = await import('./session')

    FakeBroadcastChannel.holdMessages = true
    first.setSessionUnread('stored-session', true)
    second.setSessionUnread('stored-session', false)
    FakeBroadcastChannel.flushReversed()

    expect(first.$unreadFinishedSessionIds.get()).toEqual([])
    expect(second.$unreadFinishedSessionIds.get()).toEqual([])
  })
})
