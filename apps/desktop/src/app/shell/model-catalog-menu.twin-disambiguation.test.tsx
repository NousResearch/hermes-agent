import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeAll, beforeEach, expect, it, vi } from 'vitest'

import { DropdownMenu, DropdownMenuContent } from '@/components/ui/dropdown-menu'
import { $localModelsEnabled } from '@/store/local-models-flag'
import { $localRuntimeJobs } from '@/store/local-runtime-jobs'
import { $visibleModels, emptyProviderSentinelKey, modelVisibilityKey } from '@/store/model-visibility'

import { ModelCatalogMenu, type ModelMenuController } from './model-catalog-menu'

// Two providers carrying the SAME model family under distinct catalog ids —
// the display layer strips vendor prefixes, so both rows read "GLM 5.3 Flash".
// This is the real shape that mis-billed a session: the Hugging Face twin of a
// Nous Portal model was committed from search and silently moved the spend.
const TWIN_PROVIDERS = [
  { models: ['zai-org/GLM-5.3-Flash'], name: 'Hugging Face', slug: 'huggingface' },
  { models: ['z-ai/glm-5.3-flash'], name: 'Nous Portal', slug: 'nous' }
]

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
  getGlobalModelOptions.mockResolvedValue({ providers: TWIN_PROVIDERS })
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

function renderMenu(current = { effort: '', fast: false, model: '', provider: '' }) {
  const select = vi.fn(async () => true)

  const controller: ModelMenuController = {
    applyPreset: vi.fn(),
    current,
    presetFor: () => ({}),
    select,
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

  return { select }
}

const searchBox = () => screen.getByRole('textbox', { name: 'Search models' })

// Search highlighting fragments labels into <mark> spans, so text queries must
// match on the row's assembled textContent instead of a single text node.
const glmRows = () =>
  screen.getAllByRole('menuitem').filter(row => /glm\s*5\.3\s*flash/i.test(row.textContent ?? ''))

it('tags identically-named twins with their provider so rows are distinguishable', async () => {
  renderMenu()
  await screen.findAllByText(/GLM 5\.3 Flash/i)

  // Both rows carry their owner's name inline (not only in the group header).
  const rows = glmRows()

  expect(rows.length).toBeGreaterThanOrEqual(2)
  expect(rows.some(row => /hugging face/i.test(row.textContent ?? ''))).toBe(true)
  expect(rows.some(row => /nous portal/i.test(row.textContent ?? ''))).toBe(true)
})

it('search Enter commits the user-ENABLED twin, not the alphabetically-first hidden one', async () => {
  // The user curated their picker down to the Nous copy only; the Hugging Face
  // twin is hidden (its provider carries the explicit hide-all sentinel). Search
  // still reveals it (reachability is intentional), but the enabled model must
  // win the auto-selected slot.
  $visibleModels.set(
    new Set([modelVisibilityKey('nous', 'z-ai/glm-5.3-flash'), emptyProviderSentinelKey('huggingface')])
  )

  const { select } = renderMenu()

  await screen.findAllByText(/GLM 5\.3 Flash/i)
  fireEvent.change(searchBox(), { target: { value: 'glm' } })
  await vi.waitFor(() => {
    expect(glmRows().length).toBeGreaterThanOrEqual(2)
  })
  fireEvent.keyDown(searchBox(), { key: 'Enter' })

  expect(select).toHaveBeenCalledWith('z-ai/glm-5.3-flash', 'nous')
})

it('search Enter prefers the current provider when nothing is curated', async () => {
  const { select } = renderMenu({ effort: '', fast: false, model: 'z-ai/glm-5.3-flash', provider: 'nous' })

  await screen.findAllByText(/GLM 5\.3 Flash/i)
  fireEvent.change(searchBox(), { target: { value: 'glm' } })
  await vi.waitFor(() => {
    expect(glmRows().length).toBeGreaterThanOrEqual(2)
  })
  fireEvent.keyDown(searchBox(), { key: 'Enter' })

  // The current model row is a no-op commit (menu closes); select must NOT
  // have been dispatched to the foreign twin.
  expect(select).not.toHaveBeenCalledWith('zai-org/GLM-5.3-Flash', 'huggingface')
})

it('keeps plain alphabetical order when no query is active', async () => {
  renderMenu()
  await screen.findAllByText(/GLM 5\.3 Flash/i)

  const headers = screen.getAllByText(/Hugging Face|Nous Portal/i).map((el: HTMLElement) => el.textContent)

  expect(headers.indexOf('Hugging Face')).toBeLessThan(headers.indexOf('Nous Portal'))
})
