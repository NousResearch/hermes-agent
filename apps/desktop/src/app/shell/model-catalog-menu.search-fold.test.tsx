import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

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




// The row label span (class "truncate") carries the full label text plus the
// meta suffix ("High"/"Min"), so match by prefix. Works for both the plain
// and the <mark>-split rendering.
function rowTruncateSpan(prefix: string) {
  return screen.queryByText((_, element) =>
    Boolean(element?.classList.contains('truncate') && (element?.textContent ?? '').startsWith(prefix))
  )
}

// The search filter and the highlight must agree: whatever a query reveals,
// the row's label marks the query (separator-folded equivalence). Pre-fix,
// hyphen queries revealed rows with zero marks, and space queries matched
// only via the display segment - both polarity failures covered here.
describe('model catalog search: fold between filter and highlight', () => {
  it('HYPHEN query both filters and highlights the SPACED label (the reported defect - fails pre-fix)', async () => {
    renderMenu()
    await screen.findByText(/Qwen3\.8 Flash/i)

    fireEvent.change(screen.getByRole('textbox', { name: 'Search models' }), { target: { value: 'qwen3.8-flash' } })

    await vi.waitFor(() => {
      // Whole label matches contiguously -> renders as a single <mark>.
      expect(screen.getByText('Qwen3.8 Flash', { selector: 'mark' })).toBeDefined()
    })
  })

  it('SPACE query finds the HYPHENATED id (superset guarantee) and highlights', async () => {
    renderMenu()
    await screen.findByText(/Qwen3\.8 Flash/i)

    fireEvent.change(screen.getByRole('textbox', { name: 'Search models' }), { target: { value: 'qwen3 8' } })

    await vi.waitFor(() => {
      // Partial label match -> <mark>Qwen3.8</mark> + ' Flash'.
      expect(screen.getByText('Qwen3.8', { selector: 'mark' })).toBeDefined()
      // The row label (before the meta suffix) is intact.
      expect(rowTruncateSpan('Qwen3.8 Flash')).toBeDefined()
      // The fold must not over-match: a qwen query still hides GPT rows.
      expect(rowTruncateSpan('GPT-5.1')).toBeNull()
    })
  })
})
