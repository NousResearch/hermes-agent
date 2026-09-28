import { expect, it, vi } from 'vitest'

import * as bridge from './group-chats'
import { groupChats, registerGroupChatsProvider } from './group-chats'

it('follows loading, ready, and provider disposal notifications', () => {
  let ready = false

  let changed = () => {}
  let notifications = 0
  const unsubscribe = groupChats.subscribe(() => notifications++)

  const dispose = registerGroupChatsProvider({
    status: () => (ready ? 'ready' : 'loading'),
    getSnapshot: () => ({ rooms: [] }),
    subscribe: listener => {
      changed = listener

      return () => {}
    },
    send: () => ({ accepted: true, threadId: 'thread' })
  })

  expect(groupChats.status()).toBe('loading')
  expect(groupChats.send({ roomId: 'room', text: 'hi' })).toEqual({ accepted: false, error: 'loading' })
  ready = true
  changed()
  expect(groupChats.status()).toBe('ready')
  expect(groupChats.send({ roomId: 'room', text: 'hi' })).toEqual({ accepted: true, threadId: 'thread' })
  dispose()
  expect(groupChats.status()).toBe('unavailable')
  expect(notifications).toBe(3)
  unsubscribe()
})

it('fails closed without a Bots provider', () => {
  expect(groupChats.version).toBe(1)
  expect(groupChats.status()).toBe('unavailable')
  expect(groupChats.getSnapshot()).toEqual({ rooms: [] })
  expect(groupChats.send({ roomId: 'room', text: 'hello' })).toEqual({ accepted: false, error: 'unavailable' })
})

it('isolates consumer notification failures and refuses reentrant submissions', () => {
  const error = vi.spyOn(console, 'error').mockImplementation(() => {})

  const failing = bridge.groupChats.subscribe(() => {
    throw new Error('consumer failed')
  })

  let changed!: () => void

  const sent = vi.fn(() => {
    changed()
    expect(bridge.groupChats.send({ roomId: 'room', text: 'reentrant' })).toEqual({
      accepted: false,
      error: 'submission-in-flight'
    })

    return { accepted: true as const, threadId: 'thread' }
  })

  let dispose = () => {}

  try {
    dispose = bridge.registerGroupChatsProvider({
      status: () => 'ready',
      getSnapshot: () => ({ rooms: [] }),
      subscribe: fn => {
        changed = fn

        return () => {}
      },
      send: sent
    })
    expect(bridge.groupChats.send({ roomId: 'room', text: 'hi' })).toEqual({ accepted: true, threadId: 'thread' })
    expect(sent).toHaveBeenCalledTimes(1)
  } finally {
    failing()
    dispose()
    error.mockRestore()
  }
})

it('fails closed without an engine and follows only the current provider generation', () => {
  expect(bridge.groupChats.status()).toBe('unavailable')
  expect(bridge.groupChats.send({ roomId: 'room', text: 'hi' })).toEqual({ accepted: false, error: 'unavailable' })
  const listener = vi.fn()
  const unsubscribe = bridge.groupChats.subscribe(listener)

  const provider = {
    status: () => 'ready' as const,
    getSnapshot: () => ({ rooms: [] }),
    subscribe: () => () => {},
    send: () => ({ accepted: true as const, threadId: 'thread' })
  }

  const old = bridge.registerGroupChatsProvider(provider)
  const current = bridge.registerGroupChatsProvider(provider)
  old()
  expect(bridge.groupChats.status()).toBe('ready')
  expect(bridge.groupChats.send({ roomId: 'room', text: 'hi' })).toEqual({ accepted: true, threadId: 'thread' })
  current()
  expect(bridge.groupChats.status()).toBe('unavailable')
  expect(listener).toHaveBeenCalled()
  unsubscribe()
  const count = listener.mock.calls.length
  bridge.registerGroupChatsProvider(provider)()
  expect(listener).toHaveBeenCalledTimes(count)
})

it('drops stale provider notifications and old disposal without detaching the replacement', () => {
  const notify = vi.fn()
  const unsubscribe = groupChats.subscribe(notify)

  let oldChanged = () => {}

  const provider = {
    status: () => 'ready' as const,
    getSnapshot: () => ({ rooms: [] }),
    subscribe: (listener: () => void) => {
      oldChanged = listener

      return () => {}
    },
    send: () => ({ accepted: true as const, threadId: 'thread' })
  }

  const old = registerGroupChatsProvider(provider)
  const stale = oldChanged
  const current = registerGroupChatsProvider(provider)

  try {
    notify.mockClear()
    stale()
    old()
    expect(notify).not.toHaveBeenCalled()
    expect(groupChats.status()).toBe('ready')
    oldChanged()
    expect(notify).toHaveBeenCalledTimes(1)
  } finally {
    unsubscribe()
    current()
  }
})
