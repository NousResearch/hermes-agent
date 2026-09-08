import { afterEach, describe, expect, it, vi } from 'vitest'

import { createSerialTask, startCompletionPoll } from './completion-poll'

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

  it('cancels stale publication and serializes a replacement lifecycle behind the active request', async () => {
    vi.useFakeTimers()
    const first = deferred<string>()
    const second = deferred<string>()
    const producer = vi.fn((profile: string) => (profile === 'A' ? first.promise : second.promise))
    const serial = createSerialTask(producer)
    const publishA = vi.fn()
    const publishB = vi.fn()

    const stopA = startCompletionPoll({
      delayMs: 2_000,
      poll: signal => serial('A', signal),
      publish: publishA
    })

    await Promise.resolve()
    stopA()

    const stopB = startCompletionPoll({
      delayMs: 2_000,
      poll: signal => serial('B', signal),
      publish: publishB
    })

    await Promise.resolve()
    expect(producer).toHaveBeenCalledTimes(1)

    first.resolve('stale')
    await vi.advanceTimersByTimeAsync(0)
    expect(producer).toHaveBeenCalledTimes(2)
    expect(publishA).not.toHaveBeenCalled()

    second.resolve('fresh')
    await vi.advanceTimersByTimeAsync(0)
    expect(publishB).toHaveBeenCalledWith('fresh')

    stopB()
    expect(vi.getTimerCount()).toBe(0)
  })
})
