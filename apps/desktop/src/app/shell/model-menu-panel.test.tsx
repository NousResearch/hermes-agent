import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, findAllByText, findByText, fireEvent, render } from '@testing-library/react'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import { DropdownMenu, DropdownMenuContent } from '@/components/ui/dropdown-menu'
import { $activeSessionId, $currentModel, $currentProvider, $currentReasoningEffort } from '@/store/session'

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

const ANTHROPIC_PROVIDER = {
  capabilities: {
    'anthropic/claude-sonnet-4-6': { fast: true, reasoning: true }
  },
  models: ['anthropic/claude-opus-4-8', 'anthropic/claude-sonnet-4-6'],
  name: 'Anthropic',
  slug: 'anthropic'
}

beforeEach(() => {
  $activeSessionId.set('runtime-1')
  $currentModel.set('')
  $currentProvider.set('')
  $currentReasoningEffort.set('')
  getGlobalModelOptions.mockResolvedValue({ providers: [MOA_PROVIDER] })
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

describe('ModelMenuPanel current model row', () => {
  it('pins the current model below the scrollable model list and above MoA presets', async () => {
    $currentProvider.set('anthropic')
    $currentModel.set('anthropic/claude-sonnet-4-6')
    $currentReasoningEffort.set('high')
    getGlobalModelOptions.mockResolvedValue({
      model: 'anthropic/claude-sonnet-4-6',
      provider: 'anthropic',
      providers: [ANTHROPIC_PROVIDER, MOA_PROVIDER]
    })

    renderPanel()

    await findAllByText(document.body, 'Sonnet 4 6')
    await findByText(document.body, 'Current')
    await findByText(document.body, 'MoA presets')

    const bodyText = document.body.textContent ?? ''
    const listIndex = bodyText.indexOf('Anthropic')
    const currentIndex = bodyText.indexOf('Current')
    const moaIndex = bodyText.indexOf('MoA presets')
    const currentSection = bodyText.slice(currentIndex, moaIndex)

    expect(listIndex).toBeGreaterThanOrEqual(0)
    expect(listIndex).toBeLessThan(currentIndex)
    expect(currentIndex).toBeLessThan(moaIndex)
    expect(currentSection).toContain('Sonnet 4 6')
    expect(currentSection).toContain('High')
  })
})

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
