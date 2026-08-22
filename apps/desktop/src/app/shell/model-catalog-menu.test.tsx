import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import { DropdownMenu, DropdownMenuContent } from '@/components/ui/dropdown-menu'
import {
  $modelVisibilityOpen,
  $visibleModels,
  modelVisibilityKey,
  setModelVisibilityOpen,
  setVisibleModels
} from '@/store/model-visibility'

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
  setApiRequestProfile: vi.fn()
}))

beforeEach(() => {
  $visibleModels.set(null)
  setModelVisibilityOpen(false)
  getGlobalModelOptions.mockResolvedValue({
    providers: [{ models: ['gemini-3.1-pro', 'gemini-2.5-flash'], name: 'Google', slug: 'google' }]
  })
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

// A minimal controller — these tests are about the CATALOG's own behaviour
// (what it lists, what it offers), not about what any host does with a pick.
function renderMenu() {
  const select = vi.fn()

  const controller: ModelMenuController = {
    applyPreset: vi.fn(),
    current: { effort: '', fast: false, model: '', provider: '' },
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

  return select
}

// Curation is ONE global preference, so it belongs to the catalog rather than
// to whichever surface mounted it. If a host had to opt in, the composer and
// the kanban board would end up disagreeing about what "my models" means —
// which is exactly the drift extracting this component was meant to prevent.
describe('the catalog owns model curation', () => {
  it('honours the stored Edit Models shortlist', async () => {
    setVisibleModels(new Set([modelVisibilityKey('google', 'gemini-2.5-flash')]))

    renderMenu()

    await screen.findByText(/Gemini 2\.5 Flash/i)
    expect(screen.queryByText(/Gemini 3\.1 Pro/i)).toBeNull()
  })

  it('still finds a hidden model by search — curation narrows the default view, not the catalog', async () => {
    setVisibleModels(new Set([modelVisibilityKey('google', 'gemini-2.5-flash')]))

    renderMenu()
    await screen.findByText(/Gemini 2\.5 Flash/i)

    const input = screen.getByRole('textbox', { name: 'Search models' })

    fireEvent.change(input, { target: { value: 'gemini-3.1' } })

    await vi.waitFor(() => {
      expect(screen.queryByText(/Gemini 3\.1 Pro/i)).not.toBeNull()
    })
  })

  it('offers Edit Models without the host wiring it up', async () => {
    renderMenu()
    await screen.findByText(/Gemini 3\.1 Pro/i)

    fireEvent.click(screen.getByText('Edit Models…'))

    expect($modelVisibilityOpen.get()).toBe(true)
  })
})

describe('unavailable models are inert', () => {
  it('greys out and disables a provider without usable credentials', async () => {
    getGlobalModelOptions.mockResolvedValue({
      providers: [
        {
          models: ['gemini-3.1-pro'],
          name: 'Google',
          slug: 'google',
          authenticated: false,
          warning: 'Add an API key to use Google models'
        }
      ]
    })

    const select = renderMenu()
    await screen.findByText(/Gemini 3\.1 Pro/i)

    // The provider's warning is surfaced on the row.
    expect(screen.getByText(/Add an API key/)).toBeTruthy()

    // Clicking the unauthenticated row must not commit a selection.
    fireEvent.click(screen.getByText(/Gemini 3\.1 Pro/i))
    expect(select).not.toHaveBeenCalled()
  })

  it('disables a tier-locked model but leaves its siblings selectable', async () => {
    getGlobalModelOptions.mockResolvedValue({
      providers: [
        {
          models: ['gemini-3.1-pro', 'gemini-2.5-flash'],
          name: 'Google',
          slug: 'google',
          unavailable_models: ['gemini-3.1-pro']
        }
      ]
    })

    const select = renderMenu()
    await screen.findByText(/Gemini 3\.1 Pro/i)

    // The locked model is inert…
    fireEvent.click(screen.getByText(/Gemini 3\.1 Pro/i))
    expect(select).not.toHaveBeenCalled()

    // …while the available sibling still selects normally.
    fireEvent.click(screen.getByText(/Gemini 2\.5 Flash/i))
    expect(select).toHaveBeenCalledWith('gemini-2.5-flash', 'google')
  })
})
