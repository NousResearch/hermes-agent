import { describe, expect, it, vi, beforeEach, afterEach } from 'vitest'
import { keyedTimeouts } from '@/lib/keyed-timeouts'

describe('keyedTimeouts', () => {
  beforeEach(() => vi.useFakeTimers())
  afterEach(() => vi.useRealTimers())

  it('fires callback after delay', () => {
    const kt = keyedTimeouts()
    const cb = vi.fn()
    kt.schedule('key1', 1000, cb)
    expect(cb).not.toHaveBeenCalled()
    vi.advanceTimersByTime(1000)
    expect(cb).toHaveBeenCalledTimes(1)
  })

  it('replaces previous timeout for same key', () => {
    const kt = keyedTimeouts()
    const cb = vi.fn()
    kt.schedule('key1', 1000, cb)
    kt.schedule('key1', 2000, cb)
    vi.advanceTimersByTime(1000)
    expect(cb).not.toHaveBeenCalled()
    vi.advanceTimersByTime(1000)
    expect(cb).toHaveBeenCalledTimes(1)
  })

  it('cancel removes timeout', () => {
    const kt = keyedTimeouts()
    const cb = vi.fn()
    kt.schedule('key1', 1000, cb)
    kt.cancel('key1')
    vi.advanceTimersByTime(2000)
    expect(cb).not.toHaveBeenCalled()
  })

  it('cancel on non-existent key is safe', () => {
    const kt = keyedTimeouts()
    expect(() => kt.cancel('nonexistent')).not.toThrow()
  })

  it('different keys are independent', () => {
    const kt = keyedTimeouts()
    const cbA = vi.fn()
    const cbB = vi.fn()
    kt.schedule('a', 500, cbA)
    kt.schedule('b', 1000, cbB)
    vi.advanceTimersByTime(500)
    expect(cbA).toHaveBeenCalledTimes(1)
    expect(cbB).not.toHaveBeenCalled()
    vi.advanceTimersByTime(500)
    expect(cbB).toHaveBeenCalledTimes(1)
  })
})
