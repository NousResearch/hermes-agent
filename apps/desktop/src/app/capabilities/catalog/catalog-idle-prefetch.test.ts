import { afterEach, describe, expect, it, vi } from 'vitest'

const { prefetchQuery } = vi.hoisted(() => ({ prefetchQuery: vi.fn() }))
vi.mock('@/lib/query-client', () => ({ queryClient: { prefetchQuery } }))

import { prefetchCatalogWhenIdle } from './catalog-data'

afterEach(() => {
  prefetchQuery.mockClear()
  vi.useRealTimers()
  vi.unstubAllGlobals()
})

describe('prefetchCatalogWhenIdle', () => {
  it('uses and cancels the native idle callback when available', () => {
    const request = vi.fn(() => 17)
    const cancel = vi.fn()
    vi.stubGlobal('requestIdleCallback', request)
    vi.stubGlobal('cancelIdleCallback', cancel)

    const cleanup = prefetchCatalogWhenIdle('plugins')

    expect(request).toHaveBeenCalledOnce()
    expect(request).toHaveBeenCalledWith(expect.any(Function), { timeout: 5_000 })
    cleanup()
    expect(cancel).toHaveBeenCalledExactlyOnceWith(17)
  })

  it('prefetches using a cancellable timer when idle callbacks are unavailable', () => {
    vi.useFakeTimers()
    vi.stubGlobal('requestIdleCallback', undefined)
    vi.stubGlobal('cancelIdleCallback', undefined)

    prefetchCatalogWhenIdle('plugins')

    expect(prefetchQuery).not.toHaveBeenCalled()
    vi.runOnlyPendingTimers()
    expect(prefetchQuery).toHaveBeenCalledOnce()
  })

  it('cancels the fallback timer when the view unmounts', () => {
    vi.useFakeTimers()
    vi.stubGlobal('requestIdleCallback', undefined)

    const cleanup = prefetchCatalogWhenIdle('plugins')
    cleanup()
    vi.runOnlyPendingTimers()

    expect(prefetchQuery).not.toHaveBeenCalled()
  })
})
