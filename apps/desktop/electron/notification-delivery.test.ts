import { EventEmitter } from 'node:events'

import { afterEach, expect, it, vi } from 'vitest'

import { showNativeNotification } from './notification-delivery'

afterEach(() => {
  vi.clearAllTimers()
  vi.useRealTimers()
})

it.each(['show', 'failed'] as const)('waits for native %s and cleans up delivery listeners', async event => {
  vi.useFakeTimers()
  const notification = Object.assign(new EventEmitter(), { show: vi.fn() })
  const result = showNativeNotification(notification)
  const settled = vi.fn()
  void result.then(settled)
  await Promise.resolve()
  expect(settled).not.toHaveBeenCalled()
  notification.emit(event)
  await expect(result).resolves.toBe(event === 'show')
  expect(notification.listenerCount('show')).toBe(0)
  expect(notification.listenerCount('failed')).toBe(0)
  expect(vi.getTimerCount()).toBe(0)
})

it('observes a synchronous show event and converts synchronous exceptions to failure', async () => {
  vi.useFakeTimers()

  const shown = Object.assign(new EventEmitter(), {
    show: () => {
      shown.emit('show')
    }
  })

  await expect(showNativeNotification(shown)).resolves.toBe(true)

  const failed = Object.assign(new EventEmitter(), {
    show: () => {
      throw new Error('Native presenter failed')
    }
  })

  await expect(showNativeNotification(failed)).resolves.toBe(false)
  expect(failed.listenerCount('show')).toBe(0)
  expect(failed.listenerCount('failed')).toBe(0)
  expect(vi.getTimerCount()).toBe(0)
})

it('bounds silent OS delivery and leaves activation listeners intact', async () => {
  vi.useFakeTimers()
  const notification = Object.assign(new EventEmitter(), { show: vi.fn() })
  const activated = vi.fn()
  notification.on('click', activated)
  const result = showNativeNotification(notification)
  await vi.runAllTimersAsync()
  await expect(result).resolves.toBe(false)
  expect(notification.listenerCount('show')).toBe(0)
  expect(notification.listenerCount('failed')).toBe(0)
  expect(vi.getTimerCount()).toBe(0)
  notification.emit('click')
  expect(activated).toHaveBeenCalledTimes(1)
})
