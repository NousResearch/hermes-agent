import { EventEmitter } from 'node:events'

import { afterEach, expect, it, vi } from 'vitest'

import { createNotificationRegistry } from './notification-registry'

afterEach(() => vi.useRealTimers())

it('retains a dismissed banner until its later click or action is consumed', () => {
  vi.useFakeTimers()
  const registry = createNotificationRegistry()
  for (const event of ['click', 'action']) {
    const notification = Object.assign(new EventEmitter(), { close: vi.fn() })
    const handler = vi.fn()
    notification.on(event, handler)
    registry.retain(notification)
    notification.emit('close')
    expect(registry.has(notification)).toBe(true)
    notification.emit(event)
    expect(handler).toHaveBeenCalledOnce()
    expect(registry.has(notification)).toBe(false)
  }
  expect(vi.getTimerCount()).toBe(0)
})

it('dismisses an expired notification before releasing it and releases failed delivery immediately', () => {
  vi.useFakeTimers()
  const registry = createNotificationRegistry({ ttlMs: 1000 })
  const notification = Object.assign(new EventEmitter(), { close: vi.fn(() => expect(registry.has(notification)).toBe(true)) })
  registry.retain(notification)
  vi.advanceTimersByTime(1000)
  expect(notification.close).toHaveBeenCalledOnce()
  expect(registry.has(notification)).toBe(false)

  const failed = Object.assign(new EventEmitter(), { close: vi.fn() })
  registry.retain(failed)
  failed.emit('failed')
  expect(registry.has(failed)).toBe(false)
  expect(failed.close).not.toHaveBeenCalled()
  expect(vi.getTimerCount()).toBe(0)
})
