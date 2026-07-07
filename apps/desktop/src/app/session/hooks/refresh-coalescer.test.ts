import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { createTrailingCoalescer } from './refresh-coalescer'

describe('createTrailingCoalescer', () => {
  beforeEach(() => {
    vi.useFakeTimers()
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it('collapses a burst of schedule() calls into one trailing run', async () => {
    const run = vi.fn(async () => undefined)
    const coalescer = createTrailingCoalescer(run, 1_500)

    coalescer.schedule()
    coalescer.schedule()
    coalescer.schedule()

    expect(run).not.toHaveBeenCalled()

    await vi.advanceTimersByTimeAsync(1_500)

    expect(run).toHaveBeenCalledTimes(1)
  })

  it('runs again after the window for schedules in a later burst', async () => {
    const run = vi.fn(async () => undefined)
    const coalescer = createTrailingCoalescer(run, 1_500)

    coalescer.schedule()
    await vi.advanceTimersByTimeAsync(1_500)
    expect(run).toHaveBeenCalledTimes(1)

    coalescer.schedule()
    await vi.advanceTimersByTimeAsync(1_500)
    expect(run).toHaveBeenCalledTimes(2)
  })

  it('queues at most one follow-up while a run is in flight (single-flight)', async () => {
    let release: () => void = () => undefined

    const run = vi.fn(
      () =>
        new Promise<void>(resolve => {
          release = resolve
        })
    )

    const coalescer = createTrailingCoalescer(run, 100)

    coalescer.schedule()
    await vi.advanceTimersByTimeAsync(100)
    expect(run).toHaveBeenCalledTimes(1)

    // Three more bursts land while the first run is still awaiting its fetch.
    coalescer.schedule()
    await vi.advanceTimersByTimeAsync(100)
    coalescer.schedule()
    await vi.advanceTimersByTimeAsync(100)

    expect(run).toHaveBeenCalledTimes(1)

    release()
    await vi.advanceTimersByTimeAsync(0)

    // Exactly one chained follow-up, not one per burst.
    expect(run).toHaveBeenCalledTimes(2)
  })

  it('swallows run() rejections and keeps working', async () => {
    const run = vi.fn().mockRejectedValueOnce(new Error('boom')).mockResolvedValue(undefined)
    const coalescer = createTrailingCoalescer(run, 100)

    coalescer.schedule()
    await vi.advanceTimersByTimeAsync(100)
    expect(run).toHaveBeenCalledTimes(1)

    coalescer.schedule()
    await vi.advanceTimersByTimeAsync(100)
    expect(run).toHaveBeenCalledTimes(2)
  })

  it('cancel() drops the pending timer and queued follow-up', async () => {
    const run = vi.fn(async () => undefined)
    const coalescer = createTrailingCoalescer(run, 100)

    coalescer.schedule()
    coalescer.cancel()
    await vi.advanceTimersByTimeAsync(1_000)

    expect(run).not.toHaveBeenCalled()
  })
})
