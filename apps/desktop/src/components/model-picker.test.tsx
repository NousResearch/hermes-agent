import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest'

import type { HermesGateway } from '@/hermes'
import type { ModelOptionsResponse } from '@/types/hermes'

import { ModelPickerDialog } from './model-picker'

class TestResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
}

beforeAll(() => {
  vi.stubGlobal('ResizeObserver', TestResizeObserver)
  Element.prototype.scrollIntoView = vi.fn()
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

describe('ModelPickerDialog', () => {
  it('requests and renders a refreshed model catalog', async () => {
    const initial: ModelOptionsResponse = {
      model: 'legacy-model',
      provider: 'nous',
      providers: [{ models: ['legacy-model'], name: 'Nous', slug: 'nous' }]
    }

    const refreshed: ModelOptionsResponse = {
      model: 'fresh-model',
      provider: 'nous',
      providers: [{ models: ['fresh-model'], name: 'Nous', slug: 'nous' }]
    }

    const request = vi.fn((_method: string, params: Record<string, unknown>) =>
      Promise.resolve(params.refresh === true ? refreshed : initial)
    )

    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })

    render(
      <QueryClientProvider client={client}>
        <ModelPickerDialog
          currentModel="legacy-model"
          currentProvider="nous"
          gw={{ request } as unknown as HermesGateway}
          onOpenChange={vi.fn()}
          onSelect={vi.fn()}
          open
          sessionId="session-1"
        />
      </QueryClientProvider>
    )

    expect(await screen.findByText('legacy-model')).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: 'Refresh models' }))

    await waitFor(() =>
      expect(request).toHaveBeenCalledWith('model.options', {
        explicit_only: true,
        refresh: true,
        session_id: 'session-1'
      })
    )

    expect(await screen.findByText('fresh-model')).toBeTruthy()
  })
})
