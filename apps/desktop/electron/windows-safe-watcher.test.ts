import assert from 'node:assert/strict'

import { test, vi } from 'vitest'

import { createDebouncedCallback, watchPath } from './windows-safe-watcher'

test('uses polling instead of fs.watch on Windows and unregisters the polling listener', () => {
  const watch = vi.fn()
  const watchFile = vi.fn()
  const unwatchFile = vi.fn()
  const onChange = vi.fn()

  const watcher = watchPath('C:\\plugins', onChange, {
    fsApi: { watch, watchFile, unwatchFile },
    isWindows: true
  })

  assert.equal(watch.mock.calls.length, 0)
  assert.equal(watchFile.mock.calls.length, 1)
  assert.deepEqual(watchFile.mock.calls[0]?.slice(0, 2), ['C:\\plugins', { interval: 2_000 }])
  watchFile.mock.calls[0]?.[2]({} as never, {} as never)
  assert.equal(onChange.mock.calls.length, 1)

  watcher.close()

  assert.equal(unwatchFile.mock.calls.length, 1)
  assert.equal(unwatchFile.mock.calls[0]?.[0], 'C:\\plugins')
  assert.equal(unwatchFile.mock.calls[0]?.[1], watchFile.mock.calls[0]?.[2])
})

test('keeps fs.watch on non-Windows platforms', () => {
  const close = vi.fn()
  const watch = vi.fn(() => ({ close }))
  const watchFile = vi.fn()
  const unwatchFile = vi.fn()

  const watcher = watchPath('/plugins', vi.fn(), {
    fsApi: { watch, watchFile, unwatchFile },
    isWindows: false
  })

  assert.equal(watch.mock.calls.length, 1)
  assert.equal(watchFile.mock.calls.length, 0)

  watcher.close()

  assert.equal(close.mock.calls.length, 1)
  assert.equal(unwatchFile.mock.calls.length, 0)
})

test('coalesces watcher events and cancels queued work on cleanup', () => {
  vi.useFakeTimers()
  const onChange = vi.fn()
  const notifier = createDebouncedCallback(onChange, 120)

  notifier.invoke()
  notifier.invoke()
  vi.advanceTimersByTime(119)
  assert.equal(onChange.mock.calls.length, 0)
  vi.advanceTimersByTime(1)
  assert.equal(onChange.mock.calls.length, 1)

  notifier.invoke()
  notifier.cancel()
  vi.runAllTimers()
  assert.equal(onChange.mock.calls.length, 1)
  vi.useRealTimers()
})
