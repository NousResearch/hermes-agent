import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import { DropdownMenu, DropdownMenuContent } from '@/components/ui/dropdown-menu'
import { $activeSessionId, $currentModel, $currentProvider } from '@/store/session'
import { $collapsedProviders } from '@/store/provider-collapse'

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

const DEEPSEEK_PROVIDER = {
  models: ['deepseek-v4-pro', 'deepseek-chat', 'deepseek-reasoner'],
  name: 'DeepSeek',
  slug: 'deepseek'
}

const GOOGLE_PROVIDER = {
  models: ['gemini-3.1-pro', 'gemini-2.5-flash', 'gemini-2.5-pro'],
  name: 'Google',
  slug: 'google'
}

const MOCK_PROVIDERS = [DEEPSEEK_PROVIDER, GOOGLE_PROVIDER, MOA_PROVIDER]

beforeEach(() => {
  $activeSessionId.set('runtime-1')
  $currentModel.set('')
  $currentProvider.set('')
  $collapsedProviders.set([])
  getGlobalModelOptions.mockResolvedValue({ providers: MOCK_PROVIDERS })
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

function renderPanel(onSelectModel = vi.fn()) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })

  const content = render(
    <QueryClientProvider client={client}>
      <DropdownMenu open>
        <DropdownMenuContent>
          <ModelMenuPanel onSelectModel={onSelectModel} requestGateway={vi.fn() as never} />
        </DropdownMenuContent>
      </DropdownMenu>
    </QueryClientProvider>
  )

  return { onSelectModel, content }
}

describe('ModelMenuPanel MoA presets', () => {
  it('selecting a MoA preset switches PERSISTENTLY via onSelectModel (not the one-shot dispatch)', async () => {
    const { content, onSelectModel } = renderPanel()

    // moaOptions is async (useQuery) — wait for the preset row to mount.
    const row = await content.findByText('MoA: BeastMode')
    fireEvent.click(row)

    // #54670: must route through the persistent model-switch path
    // i.e. onSelectModel with provider 'moa' (which session-scopes live-session
    // switches), NOT a one-shot command.dispatch that reverts after a turn.
    expect(onSelectModel).toHaveBeenCalledWith({ model: 'BeastMode', provider: 'moa' })
  })

  it('shows the check on the preset that matches the current moa selection', async () => {
    $currentProvider.set('moa')
    $currentModel.set('BeastMode')
    const { content } = renderPanel()

    const row = await content.findByText('MoA: BeastMode')
    // The check codicon renders as a sibling within the same row item.
    const item = row.closest('[role="menuitem"]') ?? row.parentElement
    expect(item?.querySelector('.codicon-check')).not.toBeNull()
  })

  it('keeps the virtual moa provider out of the main model groups (presets section only)', async () => {
    const { content } = renderPanel()

    await content.findByText('MoA: BeastMode')

    // The provider group header would read "Mixture of Agents"; the presets
    // section header reads "MoA presets". Only the latter should exist.
    // Radix DropdownMenu portals its content to document.body, so assert
    // against the body (not content.container) to see the rendered items.

    // eslint-disable-next-line no-restricted-globals
    expect(document.body.textContent).toContain('MoA presets')
    // eslint-disable-next-line no-restricted-globals
    expect(document.body.textContent).not.toContain('Mixture of Agents')
  })

  it('renders presets from the catalog even before a session exists', async () => {
    $activeSessionId.set('')
    const { onSelectModel, content } = renderPanel()

    const row = await content.findByText('MoA: BeastMode')
    fireEvent.click(row)

    // Pre-session picks are UI state shipped on the next session.create — the
    // row must not be disabled and must still route through onSelectModel.
    expect(onSelectModel).toHaveBeenCalledWith({ model: 'BeastMode', provider: 'moa' })
  })
})

describe('ModelMenuPanel provider collapse', () => {
  it('shows all provider models by default (none collapsed)', async () => {
    const { content } = renderPanel()

    await content.findByText('DeepSeek')
    expect(content.queryByText('Deepseek V4 Pro')).not.toBeNull()
    expect(content.queryByText('Deepseek Chat')).not.toBeNull()
  })

  it('collapses provider models when header is clicked', async () => {
    const { content } = renderPanel()

    const header = await content.findByText('DeepSeek')
    fireEvent.click(header)

    // Models should disappear but header stays
    expect(content.queryByText('Deepseek V4 Pro')).toBeNull()
    expect(content.queryByText('DeepSeek')).not.toBeNull()
  })

  it('expands provider models when header is clicked again', async () => {
    const { content } = renderPanel()

    const header = await content.findByText('DeepSeek')
    // Collapse
    fireEvent.click(header)
    expect(content.queryByText('Deepseek V4 Pro')).toBeNull()
    // Expand
    fireEvent.click(header)
    await vi.waitFor(() => {
      expect(content.queryByText('Deepseek V4 Pro')).not.toBeNull()
    })
  })

  it('auto-expands the active provider even when collapsed', async () => {
    $currentProvider.set('deepseek')
    $currentModel.set('deepseek-v4-pro')
    const { content } = renderPanel()

    const header = await content.findByText('DeepSeek')
    fireEvent.click(header)

    // Should still show models because it's the active provider
    expect(content.queryByText('Deepseek V4 Pro')).not.toBeNull()
  })

  it('bypasses collapse when search is active', async () => {
    const { content } = renderPanel()

    const header = await content.findByText('DeepSeek')
    fireEvent.click(header)
    expect(content.queryByText('Deepseek V4 Pro')).toBeNull()

    // Type in the search bar (auto-focused by DropdownMenuSearch)
    const input = screen.getByRole('textbox', { name: 'Search models' })
    expect(input).not.toBeNull()
    fireEvent.change(input, { target: { value: 'deepseek' } })

    // Should show models — search bypasses collapse
    await vi.waitFor(() => {
      expect(content.queryByText('Deepseek V4 Pro')).not.toBeNull()
    })
  })

  it('toggles collapse via keyboard Enter on header', async () => {
    const { content } = renderPanel()

    const header = await content.findByText('DeepSeek')
    // Radix DropdownMenuItem fires onSelect on Enter from the onKeyDown handler
    fireEvent.keyDown(header.closest('[role="menuitem"]') ?? header, { key: 'Enter' })

    expect(content.queryByText('Deepseek V4 Pro')).toBeNull()
  })
})
