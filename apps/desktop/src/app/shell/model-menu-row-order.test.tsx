import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import { DropdownMenu, DropdownMenuContent } from '@/components/ui/dropdown-menu'
import { $customModels } from '@/store/custom-models'
import { $visibleModels } from '@/store/model-visibility'
import { $collapsedProviders } from '@/store/provider-collapse'
import { $activeSessionId, $currentModel, $currentProvider } from '@/store/session'

import { ModelMenuPanel } from './model-menu-panel'

// Radix calls these on open; jsdom doesn't implement them.
beforeAll(() => {
  Element.prototype.scrollIntoView = vi.fn()
  Element.prototype.hasPointerCapture = vi.fn(() => false)
  Element.prototype.releasePointerCapture = vi.fn()
})

// Catalog order as a provider ships it: unrelated to the names the user reads.
const DEMO_PROVIDER = {
  models: ['zeta-10', 'ALPHA-1', 'alpha-10', 'beta-3', 'glm-5.2', 'glm-5.10'],
  name: 'Demo',
  slug: 'demo'
}

const PLAN_PROVIDER = {
  models: ['zeta-10', 'ALPHA-1', 'alpha-10', 'beta-3', 'glm-5.2', 'glm-5.10'],
  name: 'Plan',
  plan_usage: {
    featured_model_ids: ['zeta-10', 'beta-3'],
    gauges: [{ label: 'day', left_fraction: 0.5, scope: 'all', style: 'green' }],
    kind: 'usage_window'
  },
  slug: 'plan'
}

const getGlobalModelOptions = vi.fn()

vi.mock('@/hermes', () => ({
  getGlobalModelOptions: (...args: unknown[]) => getGlobalModelOptions(...args),
  setApiRequestProfile: vi.fn()
}))

beforeEach(() => {
  $activeSessionId.set('runtime-1')
  $currentModel.set('')
  $currentProvider.set('')
  $collapsedProviders.set([])
  $customModels.set([])
  $visibleModels.set(null)
  getGlobalModelOptions.mockResolvedValue({ providers: [DEMO_PROVIDER] })
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

function renderPanel() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })

  const requestGateway = vi.fn(async (method: string) => {
    if (method === 'model.options') {
      return getGlobalModelOptions()
    }

    throw new Error(`unexpected gateway method: ${method}`)
  })

  return render(
    <QueryClientProvider client={client}>
      <DropdownMenu open>
        <DropdownMenuContent>
          <ModelMenuPanel
            onFollowDefaultModel={undefined}
            onSelectModel={vi.fn()}
            requestGateway={requestGateway as never}
          />
        </DropdownMenuContent>
      </DropdownMenu>
    </QueryClientProvider>
  )
}

/** Row labels of one provider group, in paint order. The group's own header is
 *  a menuitem too, so it is dropped. */
function rowsOf(label: string): string[] {
  const group = screen.getByText(label, { exact: true }).closest('[role="group"]')

  if (!group) {
    return []
  }

  return Array.from(group.querySelectorAll('[role="menuitem"]'))
    .map(row => (row.querySelector('.truncate')?.textContent ?? '').trim())
    .filter(row => row && row !== label)
}

async function openProvider(label: string): Promise<string[]> {
  const content = renderPanel()

  await content.findByText(label, { exact: true })

  if (rowsOf(label).length === 0) {
    fireEvent.click(screen.getByText(label, { exact: true }))
  }

  await waitFor(() => expect(rowsOf(label).length).toBeGreaterThan(0))

  return rowsOf(label)
}

describe('ModelMenuPanel row order', () => {
  it('lists a provider alphabetically by the displayed name, case-insensitively and with numeric versions', async () => {
    const rows = await openProvider('Demo')

    expect(rows).toEqual(['ALPHA 1', 'Alpha 10', 'Beta 3', 'GLM 5.2', 'GLM 5.10', 'Zeta 10'])
  })

  it('keeps the active model in its alphabetical place instead of hoisting it', async () => {
    $currentModel.set('beta-3')
    $currentProvider.set('demo')

    const rows = await openProvider('Demo')

    expect(rows).toEqual(['ALPHA 1', 'Alpha 10', 'Beta 3', 'GLM 5.2', 'GLM 5.10', 'Zeta 10'])
  })

  it('paints a window plan’s featured picks first, then the rest in name order', async () => {
    getGlobalModelOptions.mockResolvedValue({ providers: [PLAN_PROVIDER] })

    const rows = await openProvider('Plan')

    expect(rows).toEqual(['Zeta 10', 'Beta 3', 'ALPHA 1', 'Alpha 10', 'GLM 5.2', 'GLM 5.10'])
  })
})
