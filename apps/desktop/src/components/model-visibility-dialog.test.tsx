import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, render, screen } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { requestModelOptions } from '@/lib/model-options'
import { $collapsedProviders } from '@/store/provider-collapse'
import { stubMenuDomApis, stubResizeObserver } from '@/test/jsdom'

import { ModelVisibilityDialog } from './model-visibility-dialog'

vi.mock('@/lib/model-options', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  requestModelOptions: vi.fn()
}))

stubMenuDomApis()
stubResizeObserver()
afterEach(() => {
  cleanup()
  vi.useRealTimers()
  vi.clearAllMocks()
})

it('updates an open Models popup from the Nous catalog without changing visibility', async () => {
  vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] })
  $collapsedProviders.set([])
  const price = { input: '$0.20', output: '$0.80', cache: '$0.02', free: false }
  const nous = { slug: 'nous', name: 'Nous Portal', models: ['vendor/model'], pricing: { 'vendor/model': price } }

  vi.mocked(requestModelOptions).mockResolvedValue({
    providers: [nous, { ...nous, slug: 'another-provider', name: 'Another provider' }]
  } as Awaited<ReturnType<typeof requestModelOptions>>)
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })

  render(
    <QueryClientProvider client={client}>
      <ModelVisibilityDialog onOpenChange={vi.fn()} onOpenProviders={vi.fn()} open />
    </QueryClientProvider>
  )
  await act(() => vi.advanceTimersByTimeAsync(10))
  expect(screen.getByText('In $0.20')).toBeTruthy()
  expect(screen.getAllByText('USD / 1M tokens')).toHaveLength(1)
  expect(screen.getByText('Cache $0.02')).toBeTruthy()
  const switches = screen.getAllByRole('switch').map(control => control.getAttribute('aria-checked'))

  vi.mocked(requestModelOptions).mockResolvedValue({
    providers: [{ ...nous, pricing: { 'vendor/model': { ...price, input: '$0.10', discount_percent: 50 } } }]
  } as Awaited<ReturnType<typeof requestModelOptions>>)
  // Advance the open editor's normal refresh cadence, without manual refresh.
  await act(() => vi.advanceTimersByTimeAsync(300_000))
  expect(screen.getByText('In $0.10')).toBeTruthy()
  expect(screen.getByText('-50%')).toBeTruthy()
  expect(screen.getAllByRole('switch')[0].getAttribute('aria-checked')).toBe(switches[0])
  expect(screen.queryByText('In $0.20')).toBeNull()
})
