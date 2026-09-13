import { QueryClient, QueryClientProvider, useQuery } from '@tanstack/react-query'
import { act, cleanup, render } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { useNousPricingRefresh } from './use-nous-pricing-refresh'

const pending = { providers: [{ slug: 'nous', name: 'Nous', models: ['vendor/model'], pricing_pending: true }] }
const warm = { providers: [{ slug: 'nous', name: 'Nous', models: ['vendor/model'] }] }

function Editor({
  fetch,
  scope = 'a',
  enabled = true
}: {
  fetch: () => Promise<typeof warm>
  scope?: string
  enabled?: boolean
}) {
  const queryKey = ['model-options', scope]
  useQuery({ queryKey, queryFn: fetch, initialData: pending, enabled })
  useNousPricingRefresh({ queryKey, enabled })

  return null
}

function client() {
  return new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity, gcTime: Infinity } } })
}

afterEach(() => {
  cleanup()
  vi.useRealTimers()
})

it('shares one bounded refresh loop, waits for slow requests and stops with the last editor', async () => {
  vi.useFakeTimers()
  const cache = client()
  let finish!: (value: typeof warm) => void

  const fetch = vi.fn<() => Promise<typeof warm>>().mockImplementationOnce(
    () =>
      new Promise(resolve => {
        finish = resolve
      })
  )

  fetch.mockResolvedValue(pending)

  const editors = (count: number) => (
    <QueryClientProvider client={cache}>
      {Array.from({ length: count }, (_, index) => (
        <Editor fetch={fetch} key={index} />
      ))}
    </QueryClientProvider>
  )

  const view = render(editors(3))

  await act(() => vi.advanceTimersByTimeAsync(1_500))
  expect(fetch).toHaveBeenCalledTimes(1)
  view.rerender(editors(2))
  // Cross the regular refresh boundary while a pending read is still in flight.
  await act(() => vi.advanceTimersByTimeAsync(600_000))
  expect(fetch).toHaveBeenCalledTimes(1)
  await act(async () => finish(pending))
  await act(() => vi.advanceTimersByTimeAsync(9 * 1_500))
  expect(fetch).toHaveBeenCalledTimes(10)
  await act(() => vi.advanceTimersByTimeAsync(299_999))
  expect(fetch).toHaveBeenCalledTimes(10)

  view.unmount()
  await act(() => vi.advanceTimersByTimeAsync(600_000))
  expect(fetch).toHaveBeenCalledTimes(10)
})

it('isolates changed scopes, retains query errors, and releases disabled or non-Nous editors', async () => {
  vi.useFakeTimers()
  const cache = client()
  let finishOld!: (value: typeof warm) => void

  const oldFetch = vi.fn<() => Promise<typeof warm>>(
    () =>
      new Promise(resolve => {
        finishOld = resolve
      })
  )

  const newFetch = vi.fn<() => Promise<typeof warm>>().mockRejectedValue(new Error('catalog unavailable'))

  const editor = (scope: string, enabled = true) => (
    <QueryClientProvider client={cache}>
      <Editor enabled={enabled} fetch={scope === 'a' ? oldFetch : newFetch} scope={scope} />
    </QueryClientProvider>
  )

  const view = render(editor('a'))
  await act(() => vi.advanceTimersByTimeAsync(1_500))
  view.rerender(editor('b'))
  await act(async () => finishOld(pending))
  await act(() => vi.advanceTimersByTimeAsync(1_500))
  expect(oldFetch).toHaveBeenCalledTimes(1)
  expect(newFetch).toHaveBeenCalledTimes(1)
  expect(cache.getQueryState(['model-options', 'b'])?.error?.message).toBe('catalog unavailable')

  view.rerender(editor('b', false))
  await act(() => vi.advanceTimersByTimeAsync(600_000))
  expect(oldFetch).toHaveBeenCalledTimes(1)
  expect(newFetch).toHaveBeenCalledTimes(1)

  cache.setQueryData(['model-options', 'b'], { providers: [{ slug: 'other', models: ['vendor/model'] }] })
  view.rerender(editor('b'))
  await act(() => vi.advanceTimersByTimeAsync(600_000))
  expect(newFetch).toHaveBeenCalledTimes(1)
})
