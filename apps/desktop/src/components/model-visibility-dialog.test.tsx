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

const getGlobalModelOptions = vi.fn()

vi.mock('@/hermes', () => ({
  getGlobalModelOptions: (...args: unknown[]) => getGlobalModelOptions(...args),
  setApiRequestProfile: vi.fn()
}))

beforeEach(() => {
  $visibleModels.set(null)
  $collapsedProviders.set([])
  getGlobalModelOptions.mockResolvedValue({
    providers: [{ models: ['b-ai/glm-5.3-flash', 'kios-ai/glm-5.3-flash'], name: 'Router', slug: 'router' }]
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
      <ModelVisibilityDialog onOpenChange={vi.fn()} onOpenProviders={vi.fn()} open />
    </QueryClientProvider>
  )
}

// The dialog is the other surface in #52442, and its switch is per-id — so the
// id has to be on the row, not only in the tooltip.
describe('an Edit Models row shows the raw id it toggles', () => {
  it('keeps two ids that share a display name apart', async () => {
    renderDialog()

    await screen.findAllByText('Glm 5.3 Flash')

    const rowFor = (id: string) => screen.getByText(id).closest('label')
    const baseRow = rowFor('b-ai/glm-5.3-flash')
    const kiosRow = rowFor('kios-ai/glm-5.3-flash')

    expect(baseRow).not.toBeNull()
    expect(kiosRow).not.toBeNull()
    expect(baseRow?.textContent).toContain('Glm 5.3 Flash')
    expect(kiosRow?.textContent).toContain('Glm 5.3 Flash')
    expect(kiosRow).not.toBe(baseRow)
  })

  // The row's filter haystack carries the id, so an id-only query keeps the row
  // it names and drops the sibling that shares its display name.
  it('filters by an id fragment that the label does not contain', async () => {
    renderDialog()

    await screen.findAllByText('Glm 5.3 Flash')

    fireEvent.change(screen.getByPlaceholderText('Search models'), { target: { value: 'kios' } })

    await vi.waitFor(() => {
      expect(screen.getByText('kios', { selector: 'mark' })).toBeDefined()
      expect(screen.queryByText('b-ai/glm-5.3-flash')).toBeNull()
    })
  })
})
