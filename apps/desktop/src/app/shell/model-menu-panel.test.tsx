import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, findByText, fireEvent, render } from '@testing-library/react'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import { DropdownMenu, DropdownMenuContent } from '@/components/ui/dropdown-menu'
import { $activeSessionId, $currentModel, $currentProvider } from '@/store/session'

import { ModelMenuPanel } from './model-menu-panel'

// Radix calls these on open; jsdom doesn't implement them.
beforeAll(() => {
  Element.prototype.scrollIntoView = vi.fn()
  Element.prototype.hasPointerCapture = vi.fn(() => false)
  Element.prototype.releasePointerCapture = vi.fn()
})

const getMoaModels = vi.fn()
const getGlobalModelOptions = vi.fn()

vi.mock('@/hermes', () => ({
  getGlobalModelOptions: (...args: unknown[]) => getGlobalModelOptions(...args),
  getMoaModels: (...args: unknown[]) => getMoaModels(...args)
}))

function moaPreset() {
  return {
    aggregator: { provider: 'deepseek', model: 'deepseek-v4-pro' },
    aggregator_temperature: 0.7,
    enabled: true,
    max_tokens: 4096,
    reference_models: [{ provider: 'zai', model: 'glm-5.2' }],
    reference_temperature: 0.7
  }
}

beforeEach(() => {
  $activeSessionId.set('runtime-1')
  $currentModel.set('')
  $currentProvider.set('')
  getGlobalModelOptions.mockResolvedValue({ providers: [] })
  getMoaModels.mockResolvedValue({
    default_preset: 'default',
    active_preset: 'default',
    presets: { default: moaPreset(), BeastMode: moaPreset() }
  })
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

function renderPanel(onSelectModel = vi.fn()) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  render(
    <QueryClientProvider client={client}>
      <DropdownMenu open>
        <DropdownMenuContent>
          <ModelMenuPanel onSelectModel={onSelectModel} requestGateway={vi.fn() as never} />
        </DropdownMenuContent>
      </DropdownMenu>
    </QueryClientProvider>
  )

  return onSelectModel
}

describe('ModelMenuPanel MoA presets', () => {
  it('selecting a MoA preset switches PERSISTENTLY via onSelectModel (not the one-shot dispatch)', async () => {
    const onSelectModel = renderPanel()

    // moaOptions is async (useQuery) — wait for the preset row to mount.
    const row = await findByText(document.body, 'MoA: BeastMode')
    fireEvent.click(row)

    // #54670: must route through the persistent model-switch path
    // (config.set model="<preset> --provider moa"), i.e. onSelectModel with
    // provider 'moa', NOT a one-shot command.dispatch that reverts after a turn.
    expect(onSelectModel).toHaveBeenCalledWith({ model: 'BeastMode', provider: 'moa' })
  })

  it('shows the check on the preset that matches the current moa selection', async () => {
    $currentProvider.set('moa')
    $currentModel.set('BeastMode')
    renderPanel()

    const row = await findByText(document.body, 'MoA: BeastMode')
    // The check codicon renders as a sibling within the same row item.
    const item = row.closest('[role="menuitem"]') ?? row.parentElement
    expect(item?.querySelector('.codicon-check')).not.toBeNull()
  })
})
