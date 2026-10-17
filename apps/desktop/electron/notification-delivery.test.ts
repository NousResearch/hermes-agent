import { EventEmitter } from 'node:events'

import { afterEach, expect, it, vi } from 'vitest'

import { showNotification } from './notification-delivery'

afterEach(() => {
  vi.clearAllTimers()
  vi.useRealTimers()
})

function notification(show = vi.fn()) {
  return Object.assign(new EventEmitter(), { show })
}

it('waits for the Windows native show event rather than accepting its void return', async () => {
  const item = notification()
  const outcome = vi.fn()
  const delivery = showNotification(item, 'win32').then(outcome)
  await Promise.resolve()
  expect(outcome).not.toHaveBeenCalled()
  item.emit('show')
  await delivery
  expect(outcome).toHaveBeenCalledWith(true)
})

it('refuses an asynchronous Windows failure without consuming click handlers', async () => {
  const item = notification()
  const click = vi.fn()
  item.on('click', click)
  const delivery = showNotification(item, 'win32')
  item.emit('failed', {}, 'synthetic private error is never returned or logged')
  expect(await delivery).toBe(false)
  expect(item.listenerCount('show')).toBe(0)
  expect(item.listenerCount('failed')).toBe(0)
  item.emit('click')
  expect(click).toHaveBeenCalledOnce()
})

it('bounds a Windows request with no outcome and ignores a late native event', async () => {
  vi.useFakeTimers()
  const item = notification()
  const delivery = showNotification(item, 'win32')
  await vi.advanceTimersByTimeAsync(5000)
  expect(await delivery).toBe(false)
  expect(item.listenerCount('show')).toBe(0)
  expect(item.listenerCount('failed')).toBe(0)
  item.emit('show')
  expect(await delivery).toBe(false)
})

it('attaches Windows listeners before show and refuses a thrown native call', async () => {
  const item = notification()
  item.show.mockImplementation(() => item.emit('show'))
  expect(await showNotification(item, 'win32')).toBe(true)
  expect(
    await showNotification(
      notification(
        vi.fn(() => {
          throw new Error('synthetic')
        })
      ),
      'win32'
    )
  ).toBe(false)
})

it('keeps the Linux promise and macOS void-return contracts unchanged', async () => {
  let resolve!: (outcome: boolean) => void

  const item = notification(
    vi.fn(
      () =>
        new Promise<boolean>(done => {
          resolve = done
        })
    )
  )

  const outcome = vi.fn()
  const delivery = showNotification(item, 'linux').then(outcome)
  await Promise.resolve()
  expect(outcome).not.toHaveBeenCalled()
  resolve(false)
  await delivery
  expect(outcome).toHaveBeenCalledWith(false)
  expect(await showNotification(notification(), 'darwin')).toBe(true)
})
