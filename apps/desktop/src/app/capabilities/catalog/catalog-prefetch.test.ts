// @vitest-environment node
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { queryClient } from '@/lib/query-client'

import { prefetchCatalogWhenIdle } from './catalog-data'

beforeEach(() => {
  vi.useFakeTimers()
  vi.spyOn(queryClient, 'prefetchQuery').mockResolvedValue(undefined)
})

afterEach(() => {
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
  vi.useRealTimers()
})

describe('catalog idle prefetch', () => {
  it('does not throw without requestIdleCallback (Safari) and defers via a timer', () => {
    vi.stubGlobal('requestIdleCallback', undefined)
    vi.stubGlobal('cancelIdleCallback', undefined)

    expect(() => prefetchCatalogWhenIdle('skills')).not.toThrow()
    expect(queryClient.prefetchQuery).not.toHaveBeenCalled()
    vi.runAllTimers()
    expect(queryClient.prefetchQuery).toHaveBeenCalledTimes(1)
  })

  it('cancels the fallback timer before prefetch starts', () => {
    vi.stubGlobal('requestIdleCallback', undefined)
    vi.stubGlobal('cancelIdleCallback', undefined)

    const cancel = prefetchCatalogWhenIdle('plugins')

    expect(vi.getTimerCount()).toBe(1)
    cancel()
    expect(vi.getTimerCount()).toBe(0)
    vi.runAllTimers()
    expect(queryClient.prefetchQuery).not.toHaveBeenCalled()
  })

  it('uses the native idle scheduler when present', () => {
    const requestIdle = vi.fn().mockReturnValue(42)
    const cancelIdle = vi.fn()

    vi.stubGlobal('requestIdleCallback', requestIdle)
    vi.stubGlobal('cancelIdleCallback', cancelIdle)

    const cancel = prefetchCatalogWhenIdle('plugins')

    expect(requestIdle).toHaveBeenCalledWith(expect.any(Function), { timeout: 5_000 })
    expect(vi.getTimerCount()).toBe(0)
    cancel()
    expect(cancelIdle).toHaveBeenCalledWith(42)
  })
})
