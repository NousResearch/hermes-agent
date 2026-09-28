import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeAll, beforeEach, expect, it, vi } from 'vitest'

import { DropdownMenu, DropdownMenuContent } from '@/components/ui/dropdown-menu'
import { $localModelsEnabled } from '@/store/local-models-flag'
import { $localRuntimeJobs } from '@/store/local-runtime-jobs'
import { $visibleModels } from '@/store/model-visibility'

import { ModelCatalogMenu, type ModelMenuController } from './model-catalog-menu'

// Radix calls these on open; jsdom doesn't implement them.
beforeAll(() => {
  Element.prototype.scrollIntoView = vi.fn()
  Element.prototype.hasPointerCapture = vi.fn(() => false)
  Element.prototype.releasePointerCapture = vi.fn()
})

const getGlobalModelOptions = vi.fn()

vi.mock('@/hermes', () => ({
  getGlobalModelOptions: (...args: unknown[]) => getGlobalModelOptions(...args),
  getLocalModelsJobs: vi.fn(async () => ({ jobs: [] })),
  getLocalModelsStatus: vi.fn().mockResolvedValue({ loading: {} }),
  setApiRequestProfile: vi.fn()
}))

beforeEach(() => {
  $visibleModels.set(null)
  $localRuntimeJobs.set([])
  $localModelsEnabled.set(false)
  getGlobalModelOptions.mockResolvedValue({
    providers: [{ models: ['qwen3.8-flash', 'gpt-5.1'], name: 'OpenRouter', slug: 'openrouter' }]
  })
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

function renderMenu() {
  const controller: ModelMenuController = {
    applyPreset: vi.fn(),
    current: { effort: '', fast: false, model: '', provider: '' },
    presetFor: () => ({}),
    select: vi.fn(async () => true),
    setOptions: vi.fn()
  }

  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })

  render(
    <QueryClientProvider client={client}>
      <DropdownMenu open>
        <DropdownMenuContent>
          <ModelCatalogMenu controller={controller} />
        </DropdownMenuContent>
      </DropdownMenu>
    </QueryClientProvider>
  )
}

it('matches and highlights separator-equivalent queries without revealing unrelated models', async () => {
  renderMenu()
  await screen.findByText(/Qwen3\.8 Flash/i)

  for (const [query, marked] of [
    ['qwen3.8-flash', 'Qwen3.8 Flash'],
    ['qwen3 8', 'Qwen3.8']
  ]) {
    fireEvent.change(screen.getByRole('textbox', { name: 'Search models' }), { target: { value: query } })
    await vi.waitFor(() => {
      expect(screen.getByText(marked, { selector: 'mark' })).not.toBeNull()
      expect(screen.queryByText(/GPT-5\.1/i)).toBeNull()
    })
  }
})
