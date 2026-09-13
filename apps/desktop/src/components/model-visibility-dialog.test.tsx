import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import { $visibleModels } from '@/store/model-visibility'
import { $collapsedProviders } from '@/store/provider-collapse'

import { ModelVisibilityDialog } from './model-visibility-dialog'

// Radix calls these on open; jsdom doesn't implement them.
beforeAll(() => {
  Element.prototype.scrollIntoView = vi.fn()
  Element.prototype.hasPointerCapture = vi.fn(() => false)
  Element.prototype.releasePointerCapture = vi.fn()
})

const requestModelOptions = vi.fn()

vi.mock('@/lib/model-options', async importOriginal => ({
  ...(await importOriginal<typeof import('@/lib/model-options')>()),
  requestModelOptions: (...args: unknown[]) => requestModelOptions(...args)
}))

const LOCAL_MODELS = ['cmd/deepseek/deepseek-v4-flash', 'cbai/deepseek-v4-flash', 'auto/best-coding']

beforeEach(() => {
  $visibleModels.set(null)
  $collapsedProviders.set([])
  requestModelOptions.mockResolvedValue({
    providers: [{ models: LOCAL_MODELS, name: 'Local', slug: 'local' }]
  })
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

function renderDialog() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })

  render(
    <QueryClientProvider client={client}>
      <ModelVisibilityDialog onOpenChange={() => {}} onOpenProviders={() => {}} open />
    </QueryClientProvider>
  )
}

describe('ModelVisibilityDialog', () => {
  it('shows the upstream in the id line, no pills', async () => {
    renderDialog()
    await screen.findByText('Best Coding')

    const names = [...document.querySelectorAll('[data-row-label]')].map(el => el.textContent)

    expect(names).toEqual(['Best Coding', 'Deepseek V4 Flash', 'Deepseek V4 Flash'])
    // No qualifier pills — the id line carries the upstream verbatim.
    expect(screen.getByText('cmd/deepseek/deepseek-v4-flash')).toBeDefined()
    expect(screen.getByText('cbai/deepseek-v4-flash')).toBeDefined()
    expect(screen.queryByText('cmd/deepseek')).toBeNull()
  })

  it('filters by upstream prefix', async () => {
    renderDialog()
    await screen.findByText('Best Coding')

    fireEvent.change(screen.getByPlaceholderText('Search models'), { target: { value: 'cbai' } })

    // Highlighting splits the id across nodes — assert on row labels instead.
    await vi.waitFor(() => {
      const names = [...document.querySelectorAll('[data-row-label]')].map(el => el.textContent)

      expect(names).toEqual(['Deepseek V4 Flash'])
    })
    expect(screen.getByText('cbai', { selector: 'mark' })).toBeDefined()
  })

  it('freezes order while toggling — no reshuffle under the pointer', async () => {
    renderDialog()
    await screen.findByText('Best Coding')

    const before = [...document.querySelectorAll('[data-row-label]')].map(el => el.textContent)
    const switches = screen.getAllByRole('switch')

    // Disable the first (enabled) row.
    fireEvent.click(switches[0])

    await vi.waitFor(() => {
      expect(switches[0].getAttribute('aria-checked')).toBe('false')
    })
    expect([...document.querySelectorAll('[data-row-label]')].map(el => el.textContent)).toEqual(before)
  })

  it('matches unordered tokens across id and upstream', async () => {
    renderDialog()
    await screen.findByText('Best Coding')

    fireEvent.change(screen.getByPlaceholderText('Search models'), { target: { value: 'flash cmd' } })

    await vi.waitFor(() => {
      const names = [...document.querySelectorAll('[data-row-label]')].map(el => el.textContent)

      expect(names).toEqual(['Deepseek V4 Flash'])
    })
    expect(screen.getAllByText('flash', { selector: 'mark' }).length).toBeGreaterThan(0)
  })

  it('caps the initial list with a show-all expander', async () => {
    const MANY = Array.from({ length: 250 }, (_, i) => `zz-model-${String(i).padStart(3, '0')}`)
    requestModelOptions.mockResolvedValue({ providers: [{ models: MANY, name: 'Many', slug: 'many' }] })
    renderDialog()
    await screen.findByText('Zz Model 000')

    expect(document.querySelectorAll('[data-row-label]').length).toBe(200)

    fireEvent.click(screen.getByText(/Show all 250/))

    await vi.waitFor(() => {
      expect(document.querySelectorAll('[data-row-label]').length).toBe(250)
    })
  })
})
