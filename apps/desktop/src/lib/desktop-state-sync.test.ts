import { describe, expect, it, vi } from 'vitest'

import { createDesktopStateSyncBus, parseCronSyncValue, parseNamedEnabledSyncValue } from './desktop-state-sync'

function fakeChannel() {
  const listeners = new Set<(event: MessageEvent) => void>()

  return {
    addEventListener: vi.fn((_type: 'message', listener: (event: MessageEvent) => void) => listeners.add(listener)),
    emit(data: unknown) {
      for (const listener of listeners) {
        listener({ data } as MessageEvent)
      }
    },
    postMessage: vi.fn(),
    removeEventListener: vi.fn((_type: 'message', listener: (event: MessageEvent) => void) =>
      listeners.delete(listener)
    )
  }
}

describe('desktop state sync bus', () => {
  it('broadcasts profile handshakes and scoped change notifications', () => {
    const channel = fakeChannel()
    const bus = createDesktopStateSyncBus(channel)

    bus.requestActiveProfile()
    bus.broadcastActiveProfile('work')
    bus.broadcastChange('model', { profile: 'work' })
    bus.broadcastChange('translucency', { value: 42 })
    bus.broadcastChange('menu-bar-transparency', { value: 35 })

    expect(channel.postMessage.mock.calls).toEqual([
      [{ type: 'active-profile-request' }],
      [{ type: 'active-profile', profile: 'work' }],
      [{ type: 'changed', domain: 'model', profile: 'work' }],
      [{ type: 'changed', domain: 'translucency', value: 42 }],
      [{ type: 'changed', domain: 'menu-bar-transparency', value: 35 }]
    ])
  })

  it('delivers messages and removes listeners cleanly', () => {
    const channel = fakeChannel()
    const bus = createDesktopStateSyncBus(channel)
    const listener = vi.fn()
    const unsubscribe = bus.subscribe(listener)

    channel.emit({ type: 'active-profile', profile: 'work' })
    expect(listener).toHaveBeenCalledWith({ type: 'active-profile', profile: 'work' })

    unsubscribe()
    channel.emit({ type: 'changed', domain: 'skills', profile: 'work' })
    expect(listener).toHaveBeenCalledTimes(1)
  })

  it('validates optimistic Desktop mirror payloads', () => {
    expect(parseNamedEnabledSyncValue({ enabled: true, name: 'research' })).toEqual({
      enabled: true,
      name: 'research'
    })
    expect(parseNamedEnabledSyncValue({ enabled: 'yes', name: 'research' })).toBeNull()
    expect(parseCronSyncValue({ enabled: false, id: 'job-1', state: 'paused' })).toEqual({
      enabled: false,
      id: 'job-1',
      state: 'paused'
    })
    expect(parseCronSyncValue({ enabled: false })).toBeNull()
  })
})
