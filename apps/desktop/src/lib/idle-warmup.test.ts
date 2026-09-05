// @vitest-environment jsdom
import { afterEach, describe, expect, it, vi } from 'vitest'

import { scheduleIdleWarmup } from './idle-warmup'

afterEach(() => {
  vi.useRealTimers()
})

describe('scheduleIdleWarmup', () => {
  it('waits for each module before scheduling the next one', async () => {
    vi.useFakeTimers()
    const order: string[] = []
    let finishFirst: () => void = () => undefined

    const first = new Promise<void>(resolve => {
      finishFirst = resolve
    })

    scheduleIdleWarmup(
      [
        async () => {
          order.push('messaging')
          await first
        },
        async () => void order.push('artifacts'),
        async () => void order.push('capabilities')
      ],
      { gapMs: 20, initialDelayMs: 100 }
    )

    await vi.advanceTimersByTimeAsync(100)
    expect(order).toEqual(['messaging'])

    await vi.advanceTimersByTimeAsync(1_000)
    expect(order).toEqual(['messaging'])

    finishFirst()
    await vi.runAllTimersAsync()
    expect(order).toEqual(['messaging', 'artifacts', 'capabilities'])
  })

  it('continues after a module fails to load', async () => {
    vi.useFakeTimers()
    const next = vi.fn(async () => undefined)

    scheduleIdleWarmup([async () => Promise.reject(new Error('chunk unavailable')), next], {
      gapMs: 20,
      initialDelayMs: 100
    })
    await vi.runAllTimersAsync()

    expect(next).toHaveBeenCalledOnce()
  })

  it('cancels work that has not started', async () => {
    vi.useFakeTimers()
    const loader = vi.fn(async () => undefined)
    const cancel = scheduleIdleWarmup([loader], { initialDelayMs: 100 })

    cancel()
    await vi.runAllTimersAsync()

    expect(loader).not.toHaveBeenCalled()
  })
})
