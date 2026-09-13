import { afterEach, expect, it, vi } from 'vitest'

import { startViewerKeepAlive } from './viewer-keep-alive'

afterEach(() => {
  vi.clearAllTimers()
  vi.useRealTimers()
})

it('renews only a live viewer, serially, and cannot restart after disposal or late liveness results', async () => {
  vi.useFakeTimers()
  let finish!: () => void
  const onKeepAlive = vi.fn(
    () =>
      new Promise<void>(resolve => {
        finish = resolve
      })
  )
  const isOpen = vi.fn(async () => true)
  const stop = startViewerKeepAlive({ isOpen, onKeepAlive })
  await vi.advanceTimersByTimeAsync(0)
  expect(onKeepAlive).toHaveBeenCalledOnce()
  await vi.advanceTimersByTimeAsync(180_000)
  expect(onKeepAlive).toHaveBeenCalledOnce()
  stop()
  finish()
  await vi.advanceTimersByTimeAsync(180_000)
  expect(onKeepAlive).toHaveBeenCalledOnce()
  expect(vi.getTimerCount()).toBe(0)

  let resolveOpen!: (open: boolean) => void
  const late = vi.fn(async () => {})
  const stopPending = startViewerKeepAlive({
    isOpen: () =>
      new Promise<boolean>(resolve => {
        resolveOpen = resolve
      }),
    onKeepAlive: late
  })
  await vi.advanceTimersByTimeAsync(0)
  stopPending()
  resolveOpen(true)
  await vi.advanceTimersByTimeAsync(180_000)
  expect(late).not.toHaveBeenCalled()
  expect(vi.getTimerCount()).toBe(0)
})

it('renews every minute, bounds failures to three attempts five seconds apart, and stops on close', async () => {
  vi.useFakeTimers()
  let open = true
  const onKeepAlive = vi.fn(async () => {})
  startViewerKeepAlive({ isOpen: () => open, onKeepAlive })
  await vi.advanceTimersByTimeAsync(0)
  expect(onKeepAlive).toHaveBeenCalledOnce()
  await vi.advanceTimersByTimeAsync(59_999)
  expect(onKeepAlive).toHaveBeenCalledOnce()
  await vi.advanceTimersByTimeAsync(1)
  expect(onKeepAlive).toHaveBeenCalledTimes(2)
  onKeepAlive.mockRejectedValueOnce(new Error('temporary'))
  await vi.advanceTimersByTimeAsync(60_000)
  await vi.advanceTimersByTimeAsync(4_999)
  expect(onKeepAlive).toHaveBeenCalledTimes(3)
  await vi.advanceTimersByTimeAsync(1)
  expect(onKeepAlive).toHaveBeenCalledTimes(4)
  onKeepAlive.mockRejectedValue(new Error('offline'))
  await vi.advanceTimersByTimeAsync(60_000)
  await vi.advanceTimersByTimeAsync(10_000)
  expect(onKeepAlive).toHaveBeenCalledTimes(7)
  await vi.advanceTimersByTimeAsync(180_000)
  expect(onKeepAlive).toHaveBeenCalledTimes(7)
  expect(vi.getTimerCount()).toBe(0)

  const closed = vi.fn(async () => {})
  startViewerKeepAlive({ isOpen: () => open, onKeepAlive: closed })
  await vi.advanceTimersByTimeAsync(0)
  open = false
  await vi.advanceTimersByTimeAsync(180_000)
  expect(closed).toHaveBeenCalledOnce()
  expect(vi.getTimerCount()).toBe(0)

  const failedProbe = vi.fn(async () => {
    throw new Error('bridge unavailable')
  })
  const unproven = vi.fn(async () => {})
  startViewerKeepAlive({ isOpen: failedProbe, onKeepAlive: unproven })
  await vi.advanceTimersByTimeAsync(180_000)
  expect(failedProbe).toHaveBeenCalledTimes(3)
  expect(unproven).not.toHaveBeenCalled()
  expect(vi.getTimerCount()).toBe(0)
})
