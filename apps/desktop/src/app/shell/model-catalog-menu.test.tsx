import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
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

// A row shows its model's effort ("Gemini 3.1 Pro  Max"), which reads as a
// fixed model+effort combo unless the row also advertises that the effort is
// editable behind it. The caret is that advertisement, and ArrowRight is the
// way in for anyone not driving the menu with a mouse.
describe('the per-row options submenu is discoverable', () => {
  it('marks each model row as opening a submenu', async () => {
    renderMenu()

    const row = await screen.findByText(/Gemini 3\.1 Pro/i)
    const trigger = row.closest('[data-slot="dropdown-menu-sub-trigger"]')

    expect(trigger).not.toBeNull()
    expect(trigger?.querySelector('.codicon-chevron-right')).not.toBeNull()
  })

  it('opens the highlighted row with ArrowRight, so effort is reachable without a mouse', async () => {
    renderMenu()
    await screen.findByText(/Gemini 3\.1 Pro/i)

    const input = screen.getByRole('textbox', { name: 'Search models' })

    // Nothing is selected yet, so highlight the first row before opening it.
    fireEvent.keyDown(input, { key: 'ArrowDown' })
    fireEvent.keyDown(input, { key: 'ArrowRight' })

    expect(await screen.findByText('Effort')).not.toBeNull()
    expect(screen.getByRole('menuitemradio', { name: 'Extra High' })).not.toBeNull()
  })

  it('returns focus to the search field when the keyboard closes the sub again', async () => {
    renderMenu()
    await screen.findByText(/Gemini 3\.1 Pro/i)

    const input = screen.getByRole('textbox', { name: 'Search models' })

    fireEvent.keyDown(input, { key: 'ArrowDown' })
    fireEvent.keyDown(input, { key: 'ArrowRight' })
    await screen.findByText('Effort')

    // ArrowLeft, not Escape: inside a sub, Escape dismisses the whole menu.
    fireEvent.keyDown(screen.getByText('Effort'), { key: 'ArrowLeft' })

    await waitFor(() => expect(input.ownerDocument.activeElement).toBe(input))
  })

  // That round trip is owed by the row we opened and by no other. Once the
  // pointer takes the menu over, Radix closes the keyboard-opened sub without
  // handing focus back — reclaiming it there would pull focus out from under
  // an interaction already in progress.
  it('leaves focus alone when the pointer takes over from a keyboard-opened sub', async () => {
    renderMenu()
    await screen.findByText(/Gemini 3\.1 Pro/i)

    const input = screen.getByRole('textbox', { name: 'Search models' })

    fireEvent.keyDown(input, { key: 'ArrowDown' })
    fireEvent.keyDown(input, { key: 'ArrowRight' })
    await screen.findByText('Effort')

    const hovered = screen.getByText(/Gemini 2\.5 Flash/i).closest('[data-slot="dropdown-menu-sub-trigger"]')

    fireEvent.pointerMove(hovered as Element, { pointerType: 'mouse' })
    await waitFor(() => expect(hovered?.getAttribute('data-state')).toBe('open'))

    expect(input.ownerDocument.activeElement).not.toBe(input)
  })

  it('leaves ArrowRight to the search field while the caret is inside the query', async () => {
    renderMenu()
    await screen.findByText(/Gemini 3\.1 Pro/i)

    const input = screen.getByRole('textbox', { name: 'Search models' }) as HTMLInputElement

    fireEvent.change(input, { target: { value: 'gemini' } })
    input.setSelectionRange(0, 0)

    fireEvent.keyDown(input, { key: 'ArrowDown' })
    fireEvent.keyDown(input, { key: 'ArrowRight' })

    expect(screen.queryByText('Effort')).toBeNull()
  })
})
