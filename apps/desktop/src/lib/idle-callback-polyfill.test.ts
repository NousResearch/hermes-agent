import { afterEach, describe, expect, it, vi } from 'vitest'

const g = globalThis as Record<string, unknown>

describe('idle-callback polyfill', () => {
  afterEach(() => {
    vi.resetModules()
    vi.useRealTimers()
  })

  it('installs a setTimeout fallback where requestIdleCallback is missing (Safari)', async () => {
    const savedRic = g.requestIdleCallback
    const savedCic = g.cancelIdleCallback
    delete g.requestIdleCallback
    delete g.cancelIdleCallback

    try {
      vi.useFakeTimers()
      await import('./idle-callback-polyfill')
      expect(typeof g.requestIdleCallback).toBe('function')

      const cb = vi.fn()
      requestIdleCallback(cb, { timeout: 5_000 })
      vi.runAllTimers()
      expect(cb).toHaveBeenCalledOnce()
      expect(cb.mock.calls[0][0].didTimeout).toBe(false)

      const cancelled = vi.fn()
      cancelIdleCallback(requestIdleCallback(cancelled))
      vi.runAllTimers()
      expect(cancelled).not.toHaveBeenCalled()
    } finally {
      g.requestIdleCallback = savedRic
      g.cancelIdleCallback = savedCic
    }
  })

  it('leaves a native implementation untouched', async () => {
    const native = vi.fn()
    const saved = g.requestIdleCallback
    g.requestIdleCallback = native

    try {
      await import('./idle-callback-polyfill')
      expect(g.requestIdleCallback).toBe(native)
    } finally {
      g.requestIdleCallback = saved
    }
  })
})
