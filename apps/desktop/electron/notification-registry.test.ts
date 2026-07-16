import assert from 'node:assert/strict'

import { test } from 'vitest'

import { createNotificationRegistry } from './notification-registry'

// A minimal fake Electron Notification: records listeners and lets a test fire
// the lifecycle events, mirroring the slice of the API the registry touches.
function makeFakeNotification() {
  const listeners = {}

  return {
    on(event, handler) {
      listeners[event] = listeners[event] || []
      listeners[event].push(handler)

      return this
    },
    emit(event, ...args) {
      for (const handler of listeners[event] || []) {
        handler(...args)
      }
    },
    listenerCount: event => (listeners[event] || []).length
  }
}

// A controllable stand-in for setTimeout/clearTimeout so the TTL backstop is
// testable without real time passing.
function makeFakeTimers() {
  const pending = new Map()
  let nextId = 1

  return {
    setTimer(fn, ms) {
      const id = nextId++

      pending.set(id, { fn, ms })

      return id
    },
    clearTimer(id) {
      pending.delete(id)
    },
    fire(id) {
      const entry = pending.get(id)

      pending.delete(id)
      entry?.fn()
    },
    fireAll() {
      for (const id of [...pending.keys()]) {
        this.fire(id)
      }
    },
    get pendingCount() {
      return pending.size
    },
    delayFor: id => pending.get(id)?.ms
  }
}

// The regression this module exists for: a shown Notification whose only
// reference was a handler-local `const` gets garbage-collected while the toast
// still sits in the Windows Action Center, taking its event emitter with it —
// so the user's click never reaches the 'click' handler. retain() must hold a
// strong reference for the whole time the toast is clickable.
test('retain keeps the notification referenced after the showing scope returns', () => {
  const timers = makeFakeTimers()
  const registry = createNotificationRegistry({ setTimer: timers.setTimer, clearTimer: timers.clearTimer })

  // Simulates the IPC handler: builds a notification, shows it, and returns
  // without keeping any reference of its own.
  const show = () => {
    const notification = makeFakeNotification()

    registry.retain(notification)

    return notification
  }

  const shown = show()

  assert.equal(registry.size, 1)
  assert.equal(registry.has(shown), true)
})

test('retain releases the notification once it is clicked', () => {
  const timers = makeFakeTimers()
  const registry = createNotificationRegistry({ setTimer: timers.setTimer, clearTimer: timers.clearTimer })
  const notification = makeFakeNotification()

  registry.retain(notification)
  notification.emit('click')

  assert.equal(registry.size, 0)
})

test('retain releases the notification once it is closed or fails', () => {
  for (const event of ['close', 'failed']) {
    const timers = makeFakeTimers()
    const registry = createNotificationRegistry({ setTimer: timers.setTimer, clearTimer: timers.clearTimer })
    const notification = makeFakeNotification()

    registry.retain(notification)
    assert.equal(registry.size, 1, `expected retention before ${event}`)

    notification.emit(event)
    assert.equal(registry.size, 0, `expected release after ${event}`)
  }
})

// Releasing must not cost the caller its own click handler: the registry adds
// listeners, it never replaces or removes them.
test('retain preserves a click handler registered by the caller', () => {
  const timers = makeFakeTimers()
  const registry = createNotificationRegistry({ setTimer: timers.setTimer, clearTimer: timers.clearTimer })
  const notification = makeFakeNotification()
  let clicked = 0

  notification.on('click', () => {
    clicked += 1
  })
  registry.retain(notification)
  notification.emit('click')

  assert.equal(clicked, 1)
  assert.equal(registry.size, 0)
})

test('a released notification is not released twice', () => {
  const timers = makeFakeTimers()
  const registry = createNotificationRegistry({ setTimer: timers.setTimer, clearTimer: timers.clearTimer })
  const first = makeFakeNotification()
  const second = makeFakeNotification()

  registry.retain(first)
  registry.retain(second)
  first.emit('click')
  first.emit('close')

  assert.equal(registry.size, 1)
  assert.equal(registry.has(second), true)
})

// Windows toasts can sit in the Action Center indefinitely and 'close' is not
// guaranteed on every platform, so a TTL backstop keeps the registry from
// growing without bound over a long session.
test('the TTL backstop releases a notification that never emits a lifecycle event', () => {
  const timers = makeFakeTimers()
  const registry = createNotificationRegistry({ ttlMs: 1000, setTimer: timers.setTimer, clearTimer: timers.clearTimer })
  const notification = makeFakeNotification()

  registry.retain(notification)
  assert.equal(registry.size, 1)

  timers.fireAll()
  assert.equal(registry.size, 0)
})

test('the TTL timer is cancelled when the notification is released early', () => {
  const timers = makeFakeTimers()
  const registry = createNotificationRegistry({ ttlMs: 1000, setTimer: timers.setTimer, clearTimer: timers.clearTimer })
  const notification = makeFakeNotification()

  registry.retain(notification)
  assert.equal(timers.pendingCount, 1)

  notification.emit('click')

  assert.equal(timers.pendingCount, 0)
})

test('retain tolerates a missing notification', () => {
  const registry = createNotificationRegistry()

  assert.equal(registry.retain(null), null)
  assert.equal(registry.size, 0)
})

test('retain returns the notification so callers can chain show()', () => {
  const timers = makeFakeTimers()
  const registry = createNotificationRegistry({ setTimer: timers.setTimer, clearTimer: timers.clearTimer })
  const notification = makeFakeNotification()

  assert.equal(registry.retain(notification), notification)
})
