import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeAll, beforeEach, expect, it, vi } from 'vitest'

import { DropdownMenu, DropdownMenuContent } from '@/components/ui/dropdown-menu'
import { $localModelsEnabled } from '@/store/local-models-flag'
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
  $localModelsEnabled.set(false)
  getGlobalModelOptions.mockResolvedValue({
    providers: [{ models: ['qwen3.8-flash', 'gpt-5.1'], name: 'OpenRouter', slug: 'openrouter' }]
  })
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

it('labels aliases with the model id while leaving unaliased models unchanged', async () => {
  getGlobalModelOptions.mockResolvedValue({
    providers: [
      {
        model_aliases: { 'aliased-model': 'quick-alias', 'injected-model': 'extra-alias' },
        models: ['aliased-model', 'plain-model', 'injected-model'],
        name: 'Picker Local',
        slug: 'picker-local'
      }
    ]
  })
  renderMenu()

  const alias = await screen.findByText('quick-alias')
  expect(alias.closest('[role="menuitem"]')?.textContent).toContain('quick-alias (aliased-model)')
  expect(screen.getByText('extra-alias').closest('[role="menuitem"]')?.textContent).toContain(
    'extra-alias (injected-model)'
  )
  expect(screen.getByText('Plain Model')).toBeDefined()
})

it('finds an aliased model by alias and still selects its canonical id', async () => {
  getGlobalModelOptions.mockResolvedValue({
    providers: [
      {
        model_aliases: { 'aliased-model': 'quick-alias' },
        models: ['aliased-model', 'plain-model'],
        name: 'Picker Local',
        slug: 'picker-local'
      }
    ]
  })
  const select = renderMenu()
  await screen.findByText('Plain Model')

  fireEvent.change(screen.getByRole('textbox', { name: 'Search models' }), { target: { value: 'quick-alias' } })
  const alias = await screen.findByText('quick-alias', { selector: 'mark' })
  expect(screen.queryByText('Plain Model')).toBeNull()
  fireEvent.click(alias.closest('[role="menuitem"]')!)
  expect(select).toHaveBeenCalledWith('aliased-model', 'picker-local')
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

  return controller.select
}

it('matches and highlights separator-equivalent queries without revealing unrelated models', async () => {
  renderMenu()
  await screen.findByText('Qwen3.8')

  // The `-flash` suffix renders as the row's variant tag, so an id-style
  // query still finds the row through the id fold even though the label no
  // longer spells the word — nothing is marked, the row stays visible.
  fireEvent.change(screen.getByRole('textbox', { name: 'Search models' }), { target: { value: 'qwen3.8-flash' } })
  await vi.waitFor(() => {
    expect(screen.getByText('Qwen3.8')).toBeDefined()
    expect(screen.queryByText(/GPT-5\.1/i)).toBeNull()
  })

  // A separator-folded query highlights the name span itself.
  fireEvent.change(screen.getByRole('textbox', { name: 'Search models' }), { target: { value: 'qwen3 8' } })
  await vi.waitFor(() => {
    expect(screen.getByText('Qwen3.8', { selector: 'mark' })).not.toBeNull()
    expect(screen.queryByText(/GPT-5\.1/i)).toBeNull()
  })
})
