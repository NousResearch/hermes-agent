import { EventEmitter } from 'node:events'
import { createRequire } from 'node:module'

import { expect, it, vi } from 'vitest'

const { observeProcessClose } = createRequire(import.meta.url)('../tests/install/e2e-assets/process-close.cjs')

it('waits for native close, not exit, and retains a close observed before hand-off', async () => {
  const pipe = { destroy: vi.fn() }
  const child = Object.assign(new EventEmitter(), { stdio: [null, pipe], exitCode: null, signalCode: null })
  const waitForClose = observeProcessClose(child)
  expect(pipe.destroy).not.toHaveBeenCalled()
  let finished = false
  const completion = waitForClose().then(() => { finished = true })
  child.emit('exit', 0)
  expect(pipe.destroy).toHaveBeenCalledOnce()
  await Promise.resolve()
  expect(finished).toBe(false)
  child.emit('close', 0)
  await completion
  expect(finished).toBe(true)
  await expect(waitForClose()).resolves.toBeUndefined()
})

it('fails if the launched process never closes', async () => {
  vi.useFakeTimers()

  try {
    const waitForClose = observeProcessClose(Object.assign(new EventEmitter(), { stdio: [], exitCode: null, signalCode: null }))
    const completion = expect(waitForClose(2_000)).rejects.toThrow('Electron process did not close')
    await vi.advanceTimersByTimeAsync(2_000)
    await completion
    expect(vi.getTimerCount()).toBe(0)
  } finally {
    vi.useRealTimers()
  }
})
