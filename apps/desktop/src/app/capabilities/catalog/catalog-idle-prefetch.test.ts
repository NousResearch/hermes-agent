import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { queryClient } from '@/lib/query-client'

import { prefetchCatalogWhenIdle } from './catalog-data'

beforeEach(() => {
  vi.useFakeTimers()
  queryClient.clear()
  vi.stubGlobal(
    'fetch',
    vi.fn(async () => ({ ok: true, json: async () => [] }))
  )
})

afterEach(() => {
  queryClient.clear()
  vi.unstubAllGlobals()
  vi.useRealTimers()
})

describe.each(['without idle callbacks', 'with idle callbacks'])('catalog prefetch %s', browser => {
  beforeEach(() => {
    vi.stubGlobal('requestIdleCallback', undefined)
    vi.stubGlobal('cancelIdleCallback', undefined)

    if (browser === 'with idle callbacks') {
      vi.stubGlobal('requestIdleCallback', (callback: () => void) => setTimeout(callback, 10))
      vi.stubGlobal('cancelIdleCallback', (id: ReturnType<typeof setTimeout>) => clearTimeout(id))
    }
  })

  it.each(['skills', 'plugins'] as const)('defers the %s fetch and warms the real query cache', async kind => {
    const cancel = prefetchCatalogWhenIdle(kind)

    expect(fetch).not.toHaveBeenCalled()
    await vi.advanceTimersByTimeAsync(10)

    expect(fetch).toHaveBeenCalledExactlyOnceWith(
      `https://nousresearch.github.io/hermes-agent/docs/api/${kind}.json`,
      expect.objectContaining({ credentials: 'omit', signal: expect.any(AbortSignal) })
    )
    expect(queryClient.getQueryData(['public-catalog', kind])).toEqual([])
    cancel()
  })

  it.each(['skills', 'plugins'] as const)('cancels the pending %s prefetch on cleanup', async kind => {
    const cancel = prefetchCatalogWhenIdle(kind)

    cancel()
    await vi.advanceTimersByTimeAsync(10)

    expect(fetch).not.toHaveBeenCalled()
    expect(queryClient.getQueryState(['public-catalog', kind])).toBeUndefined()
  })
})
