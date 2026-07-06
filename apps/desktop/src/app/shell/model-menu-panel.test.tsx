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

const getGlobalModelOptions = vi.fn()

vi.mock('@/hermes', () => ({
  getGlobalModelOptions: (...args: unknown[]) => getGlobalModelOptions(...args)
}))

// MoA presets now arrive as the catalog's virtual `moa` provider row (the same
// payload a remote gateway's model.options returns), not the /api/model/moa
// REST config.
const MOA_PROVIDER = { models: ['default', 'BeastMode'], name: 'Mixture of Agents', slug: 'moa' }

beforeEach(() => {
  $activeSessionId.set('runtime-1')
  $currentModel.set('')
  $currentProvider.set('')
  getGlobalModelOptions.mockResolvedValue({ providers: [MOA_PROVIDER] })
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

  it('keeps the virtual moa provider out of the main model groups (presets section only)', async () => {
    renderPanel()

    await findByText(document.body, 'MoA: BeastMode')

    // The provider group header would read "Mixture of Agents"; the presets
    // section header reads "MoA presets". Only the latter should exist.
    expect(document.body.textContent).toContain('MoA presets')
    expect(document.body.textContent).not.toContain('Mixture of Agents')
  })

  it('renders presets from the catalog even before a session exists', async () => {
    $activeSessionId.set('')
    const onSelectModel = renderPanel()

    const row = await findByText(document.body, 'MoA: BeastMode')
    fireEvent.click(row)

    // Pre-session picks are UI state shipped on the next session.create — the
    // row must not be disabled and must still route through onSelectModel.
    expect(onSelectModel).toHaveBeenCalledWith({ model: 'BeastMode', provider: 'moa' })
  })
})

describe('ModelMenuPanel custom providers', () => {
  it('renders the active bare custom endpoint and each named custom provider', async () => {
    getGlobalModelOptions.mockResolvedValueOnce({
      model: 'Qwen3.6-27B-NVFP4-MTP-GGUF.gguf',
      provider: 'custom',
      providers: [
        {
          api_url: 'http://192.168.50.124:8090/v1',
          is_current: true,
          is_user_defined: true,
          models: ['Qwen3.6-27B-NVFP4-MTP-GGUF.gguf'],
          name: 'Custom endpoint',
          slug: 'custom',
          source: 'model-config'
        },
        {
          api_url: 'http://192.168.50.130:8080/v1',
          is_user_defined: true,
          models: ['gemma-4-12b-omni'],
          name: '3090-gemma4-12b',
          slug: 'custom:3090-gemma4-12b',
          source: 'user-config'
        },
        {
          api_url: 'http://192.168.50.125:8000/v1',
          is_user_defined: true,
          models: ['gemma-4-26b-a4b-nvfp4'],
          name: 'spark-gemma4-26b-a4b',
          slug: 'custom:spark-gemma4-26b-a4b',
          source: 'user-config'
        }
      ]
    })

    renderPanel()

    await findByText(document.body, 'Custom endpoint')
    await findByText(document.body, '3090-gemma4-12b')
    await findByText(document.body, 'spark-gemma4-26b-a4b')
    expect(document.body.textContent).toContain('Qwen3.6 27B NVFP4 MTP GGUF.Gguf')
    expect(document.body.textContent).toContain('Gemma 4 12b Omni')
    expect(document.body.textContent).toContain('Gemma 4 26b A4b Nvfp4')
  })
})
