import { afterEach, describe, expect, it, vi } from 'vitest'

import { rafCoalesce } from './raf-coalesce'

afterEach(() => {
  vi.useRealTimers()
})

describe('rafCoalesce', () => {
  it('applies only the last pushed value, once per frame', () => {
    vi.useFakeTimers({ toFake: ['requestAnimationFrame', 'cancelAnimationFrame'] })
    const apply = vi.fn()
    const coalesced = rafCoalesce<number>(apply)

    coalesced.push(1)
    coalesced.push(2)
    vi.advanceTimersToNextFrame()

    expect(apply).toHaveBeenCalledTimes(1)
    expect(apply).toHaveBeenCalledWith(2)
  })

  it('cancel drops the pending value without applying it', () => {
    vi.useFakeTimers({ toFake: ['requestAnimationFrame', 'cancelAnimationFrame'] })
    const apply = vi.fn()
    const coalesced = rafCoalesce<number>(apply)

    coalesced.push(1)
    coalesced.cancel()
    vi.advanceTimersToNextFrame()

    expect(apply).not.toHaveBeenCalled()
  })
})
