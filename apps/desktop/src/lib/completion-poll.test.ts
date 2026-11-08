import { afterEach, describe, expect, it, vi } from 'vitest'

import { startCompletionPoll } from './completion-poll'

function deferred<T>() {
  let resolve!: (value: T) => void

  const promise = new Promise<T>(done => {
    resolve = done
  })

  return { promise, resolve }
}

describe('startCompletionPoll', () => {
  afterEach(() => {
    vi.useRealTimers()
  })

  it('keeps a slow producer single-flight and resumes cadence after completion', async () => {
    vi.useFakeTimers()
    const first = deferred<string>()
    const poll = vi.fn(() => first.promise)
    const publish = vi.fn()

    const stop = startCompletionPoll({ delayMs: 2_000, poll, publish })
    await vi.advanceTimersByTimeAsync(20_000)
    expect(poll).toHaveBeenCalledTimes(1)

    first.resolve('tail')
    await Promise.resolve()
    expect(publish).toHaveBeenCalledWith('tail')
    await vi.advanceTimersByTimeAsync(1_999)
    expect(poll).toHaveBeenCalledTimes(1)
    await vi.advanceTimersByTimeAsync(1)
    expect(poll).toHaveBeenCalledTimes(2)
    stop()
  })

  it('stop suppresses a late publish and leaves no timer behind', async () => {
    vi.useFakeTimers()
    const first = deferred<string>()
    const publish = vi.fn()

    const stop = startCompletionPoll({ delayMs: 2_000, poll: () => first.promise, publish })
    await Promise.resolve()
    stop()

    first.resolve('stale')
    await vi.advanceTimersByTimeAsync(0)
    expect(publish).not.toHaveBeenCalled()
    expect(vi.getTimerCount()).toBe(0)
  })
})
