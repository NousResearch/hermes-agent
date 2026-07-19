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
    const completion = first.createSessionCompletionToken('delayed-completion', 1)

    FakeBroadcastChannel.holdMessages = true
    first.recordSessionCompletion('stored-session', completion, true)
    second.acknowledgeSessionCompletion('stored-session', completion)
    FakeBroadcastChannel.flushReversed()

    expect(first.$unreadFinishedSessionIds.get()).toEqual([])
    expect(second.$unreadFinishedSessionIds.get()).toEqual([])
  })

  it('does not let an acknowledgement for one completion clear the next completion', async () => {
    const session = await import('./session')
    const first = session.createSessionCompletionToken('completion-1', 1)
    const second = session.createSessionCompletionToken('completion-2', 2)

    session.recordSessionCompletion('stored-session', first, true)
    session.recordSessionCompletion('stored-session', second, true)
    session.acknowledgeSessionCompletion('stored-session', first)

    expect(session.$unreadFinishedSessionIds.get()).toEqual(['stored-session'])
  })

  it('keeps an exact acknowledgement monotonic across duplicate renderer events', async () => {
    const first = await import('./session')
    vi.resetModules()
    const second = await import('./session')
    const completion = first.createSessionCompletionToken('shared-completion', 1)

    first.recordSessionCompletion('stored-session', completion, true)
    second.acknowledgeSessionCompletion('stored-session', completion)
    first.recordSessionCompletion('stored-session', completion, true)

    expect(first.$unreadFinishedSessionIds.get()).toEqual([])
    expect(second.$unreadFinishedSessionIds.get()).toEqual([])
  })

  it('rejects a delayed completion captured before a gateway reset', async () => {
    const session = await import('./session')
    const stale = session.createSessionCompletionToken('old-gateway-completion', 1)

    session.clearAllSessionUnread()
    session.recordSessionCompletion('old-backend-session', stale, true)

    expect(session.$unreadFinishedSessionIds.get()).toEqual([])

    const current = session.createSessionCompletionToken('new-gateway-completion', 2)
    session.recordSessionCompletion('new-backend-session', current, true)
    expect(session.$unreadFinishedSessionIds.get()).toEqual(['new-backend-session'])
  })
})
