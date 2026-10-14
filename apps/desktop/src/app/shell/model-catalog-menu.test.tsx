vi.mock('@/store/profile', async (): Promise<object> => {
  const { atom } = await import('nanostores')

  return { $activeGatewayProfile: atom<string>('default') }
})
vi.mock('@/store/session', async (): Promise<object> => {
  const { atom } = await import('nanostores')

  return { $connection: atom(null), $defaultReasoningEffort: atom<string>('') }
})

import type { QueryClient } from '@tanstack/react-query'
import { QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import { DropdownMenu, DropdownMenuContent } from '@/components/ui/dropdown-menu'
import { registry } from '@/contrib/registry'
import { queryClient } from '@/lib/query-client'
import { $favoriteModels, favoriteModelKey, toggleFavoriteModel } from '@/store/favorite-models'
import { $localModelsEnabled } from '@/store/local-models-flag'
import { localModelsKey, localModelsOwner } from '@/store/local-runtime-jobs'
import { setShowModelPricing } from '@/store/model-pricing'
import {
  $modelVisibilityOpen,
  $visibleModels,
  modelVisibilityKey,
  setModelVisibilityOpen,
  setVisibleModels
} from '@/store/model-visibility'
import { $defaultReasoningEffort } from '@/store/session'
import type { LocalRuntimeJob } from '@/types/hermes'

import { ModelCatalogMenu, ModelMenuCloseContext, type ModelMenuController } from './model-catalog-menu'
import { MODEL_MENU_ROW_AREA, type ModelMenuRowContribution } from './model-menu-row-decorations'

// Radix calls these on open; jsdom doesn't implement them.
beforeAll(() => {
  Element.prototype.scrollIntoView = vi.fn()
  Element.prototype.hasPointerCapture = vi.fn(() => false)
  Element.prototype.releasePointerCapture = vi.fn()
})

const getGlobalModelOptions = vi.fn()
const closeMenu = vi.fn()

vi.mock('@/hermes', () => ({
  getGlobalModelOptions: (...args: unknown[]) => getGlobalModelOptions(...args),
  // The menu kicks the app-level job poller on mount; echo the store so a
  // poll can't wipe the jobs a test staged (the real backend is authority,
  // and here the store plays that part).
  getLocalModelsJobs: vi.fn(async () => {
    const { localModelsKey, localModelsOwner } = await import('@/store/local-runtime-jobs')
    const { queryClient } = await import('@/lib/query-client')

    return {
      jobs: [
        ...(queryClient.getQueryData<readonly LocalRuntimeJob[]>(localModelsKey(localModelsOwner(), 'jobs')) ?? [])
      ]
    }
  }),
  getLocalModelsStatus: vi.fn().mockResolvedValue({ loading: {} }),
  setApiRequestProfile: vi.fn()
}))

beforeEach((): void => {
  queryClient.clear()
  queryClient.setDefaultOptions({ queries: { ...queryClient.getDefaultOptions().queries, retry: false } })
  window.localStorage.clear()
  $visibleModels.set(null)
  $favoriteModels.set([])
  queryClient.setQueryData(localModelsKey(localModelsOwner(), 'jobs'), [])
  // These suites exercise the local-models rows, which ship behind --local.
  $localModelsEnabled.set(true)
  setModelVisibilityOpen(false)
  getGlobalModelOptions.mockResolvedValue({
    providers: [{ models: ['gemini-3.1-pro', 'gemini-2.5-flash'], name: 'Google', slug: 'google' }]
  })
})

afterEach(() => {
  cleanup()
  queryClient.clear()
  // The backend mock echoes this snapshot; retire fixture jobs before jsdom
  // disappears so an in-flight app-level poll cannot schedule another tick.
  queryClient.setQueryData(localModelsKey(localModelsOwner(), 'jobs'), [])
  $defaultReasoningEffort.set('')
  vi.clearAllMocks()
})

describe('model menu row decorations (MODEL_MENU_ROW_AREA)', () => {
  it('paints a contributed icon and badge in the row slots, skipping a throwing decorator', async () => {
    const dispose = [
      registry.register({
        area: MODEL_MENU_ROW_AREA,
        data: {
          decorate: () => {
            throw new Error('broken plugin')
          }
        } satisfies ModelMenuRowContribution,
        id: 'broken'
      }),
      registry.register({
        area: MODEL_MENU_ROW_AREA,
        data: {
          decorate: ({ model, provider }) =>
            model === 'gemini-3.1-pro' ? { badge: 'new', icon: <img alt="" data-testid={`mark-${provider}`} /> } : null
        } satisfies ModelMenuRowContribution,
        id: 'marks'
      })
    ]

    try {
      renderMenu()

      const row = (await screen.findByText('Gemini 3.1 Pro')).closest('[role="menuitem"]')!
      const icon = row.querySelector('[data-slot="model-menu-row-icon"]')

      expect(icon?.querySelector('[data-testid="mark-google"]')).toBeTruthy()
      expect(row.querySelector('[data-model-menu-row-badge]')?.textContent).toBe('new')

      // A decorator returning null leaves its row bare.
      const bare = screen.getByText('Gemini 2.5').closest('[role="menuitem"]')!

      expect(bare.querySelector('[data-slot="model-menu-row-icon"]')).toBeNull()
      expect(bare.querySelector('[data-model-menu-row-badge]')).toBeNull()
    } finally {
      dispose.forEach(release => release())
    }
  })
})

describe('the current row effort', () => {
  it('does not label the current model with the profile default before its session reports one (#79807)', async () => {
    $defaultReasoningEffort.set('ultra')
    renderMenu({ effortPending: true, model: 'gemini-2.5-flash', provider: 'google' })

    const row = (await screen.findByText('Gemini 2.5')).closest('[role="menuitem"]')!

    expect(row.textContent).not.toContain('Ultra')
    cleanup()

    renderMenu({ model: 'gemini-2.5-flash', provider: 'google' })

    const settled = (await screen.findByText('Gemini 2.5')).closest('[role="menuitem"]')!

    expect(settled.textContent).toContain('Ultra')
  })
})

describe('the reasoning-effort badge (#51833)', () => {
  it('renders the effort as its own Badge chip beside the name, never inside it', async () => {
    renderMenu({ effort: 'high', model: 'gemini-2.5-flash', provider: 'google' })

    // The effort chip renders exactly "High" in its own Badge…
    const badge = await screen.findByText('High')

    expect(badge.textContent).toBe('High')
    expect(badge.getAttribute('data-slot')).toBe('badge')

    // …as a SIBLING of the truncating model-name span, so it can never read as
    // part of a differently-named model. The `-flash` variant tag is its own
    // chip between them (#118083); the name itself stays free of both.
    const nameSpan = badge.parentElement?.querySelector('.truncate')

    expect(nameSpan?.className).toContain('truncate')
    expect(nameSpan?.contains(badge)).toBe(false)
    expect(nameSpan?.textContent?.toLowerCase()).toContain('gemini 2.5')
    expect(nameSpan?.textContent?.toLowerCase()).not.toContain('flash')
    expect(nameSpan?.textContent?.toLowerCase()).not.toContain('high')
  })

  it('drops the effort badge entirely when the model has no reasoning support', async () => {
    getGlobalModelOptions.mockResolvedValue({
      providers: [
        {
          name: 'Google',
          slug: 'google',
          models: ['gemini-2.5-flash'],
          capabilities: { 'gemini-2.5-flash': { fast: false, reasoning: false } }
        }
      ]
    })

    renderMenu({ effort: 'high', model: 'gemini-2.5-flash', provider: 'google' })

    await screen.findByText('Gemini 2.5')

    await waitFor(() => {
      expect(screen.queryByText('High')).toBeNull()
      expect(screen.queryByText('Med')).toBeNull()
    })
  })
})

// A minimal controller — these tests are about the CATALOG's own behaviour
// (what it lists, what it offers), not about what any host does with a pick.
function renderMenu(current: Partial<ModelMenuController['current']> = {}) {
  const select = vi.fn()

  const controller: ModelMenuController = {
    applyPreset: vi.fn(),
    current: { effort: '', fast: false, model: '', provider: '', ...current },
    presetFor: () => ({}),
    select,
    setOptions: vi.fn()
  }

  const client: QueryClient = queryClient

  render(
    <QueryClientProvider client={client}>
      <ModelMenuCloseContext.Provider value={closeMenu}>
        <DropdownMenu open>
          <DropdownMenuContent>
            <ModelCatalogMenu controller={controller} />
          </DropdownMenuContent>
        </DropdownMenu>
      </ModelMenuCloseContext.Provider>
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

    await screen.findByText('Gemini 2.5')
    expect(screen.queryByText(/Gemini 3\.1 Pro/i)).toBeNull()
  })

  it('still finds a hidden model by search — curation narrows the default view, not the catalog', async () => {
    setVisibleModels(new Set([modelVisibilityKey('google', 'gemini-2.5-flash')]))

    renderMenu()
    await screen.findByText('Gemini 2.5')

    const input = screen.getByRole('textbox', { name: 'Search models' })

    fireEvent.change(input, { target: { value: 'gemini-3.1' } })

    await vi.waitFor(() => {
      // The fold makes this id-style query highlight the spaced label: the
      // row renders as <mark>Gemini 3.1</mark> + ' Pro'.
      expect(screen.getByText('Gemini 3.1', { selector: 'mark' })).toBeDefined()
    })
  })

  it('offers Edit Models without the host wiring it up', async () => {
    renderMenu()
    await screen.findByText(/Gemini 3\.1 Pro/i)

    fireEvent.click(screen.getByText('Edit models…'))

    expect($modelVisibilityOpen.get()).toBe(true)
  })
})

// A star is a promise about the LIST: "keep this one where I can always reach
// it". That promise is what decides where a favorite paints — its own section
// at the top, and nowhere twice.
describe('the catalog owns favorite models', () => {
  it('lifts a favorite into the Favorites section above the provider groups', async () => {
    toggleFavoriteModel('google', 'gemini-2.5-flash')

    renderMenu()

    await screen.findByText(/Gemini 2\.5/i)

    const label = screen.getByText('Favorites')
    const googleHeading = screen.getByText('Google')

    expect(label.compareDocumentPosition(googleHeading) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
  })

  it('names each provider once over its favorites when the section mixes providers', async () => {
    getGlobalModelOptions.mockResolvedValue({
      providers: [
        { models: ['gemini-3.1-pro', 'gemini-2.5-flash'], name: 'Google', slug: 'google' },
        { models: ['gemini-3.1-pro'], name: 'OpenRouter', slug: 'openrouter' }
      ]
    })
    // Starred out of provider order: the section still gathers each
    // provider's favorites under one label, in the order they were starred.
    toggleFavoriteModel('google', 'gemini-3.1-pro')
    toggleFavoriteModel('openrouter', 'gemini-3.1-pro')
    toggleFavoriteModel('google', 'gemini-2.5-flash')

    renderMenu()

    await screen.findByText('Favorites')

    const rows = screen.getAllByText(/Gemini (3\.1|2\.5)/).map(node => node.textContent)

    // One label per provider, never one per row, and each provider's
    // favorites sit together under it.
    expect(screen.getAllByText('Google')).toHaveLength(1)
    expect(screen.getAllByText('OpenRouter')).toHaveLength(1)
    expect(rows).toEqual(['Gemini 3.1 Pro', 'Gemini 2.5', 'Gemini 3.1 Pro'])
  })

  it('does not label the provider when every favorite shares one', async () => {
    toggleFavoriteModel('google', 'gemini-2.5-flash')

    renderMenu()

    await screen.findByText('Favorites')

    // Only Google's own group heading, never a label inside Favorites.
    expect(screen.getAllByText('Google')).toHaveLength(1)
  })

  it('does not also list a favorite under its provider', async () => {
    toggleFavoriteModel('google', 'gemini-2.5-flash')

    renderMenu()

    await screen.findByText('Favorites')

    // Listed once, under Favorites — not also down in Google's group.
    expect(screen.getAllByText(/Gemini 2\.5/i)).toHaveLength(1)
  })

  it('keeps a favorite whose provider is not connected without painting an empty section', async () => {
    $favoriteModels.set([favoriteModelKey('anthropic', 'claude-sonnet-4.6')])

    renderMenu()

    await screen.findByText(/Gemini 3\.1 Pro/i)
    expect(screen.queryByText('Favorites')).toBeNull()
  })

  // Curation and favorites are different questions: "which models do I
  // usually want listed" vs "which one do I want first". A star wins.
  it('shows a favorite the Edit Models shortlist hides', async () => {
    setVisibleModels(new Set([modelVisibilityKey('google', 'gemini-3.1-pro')]))
    toggleFavoriteModel('google', 'gemini-2.5-flash')

    renderMenu()

    await screen.findByText('Favorites')
    expect(screen.getAllByText(/Gemini 2\.5/i)).toHaveLength(1)
  })

  it('folds the section away while searching and lists the match in its provider place', async () => {
    toggleFavoriteModel('google', 'gemini-2.5-flash')

    renderMenu()
    await screen.findByText('Favorites')

    fireEvent.change(screen.getByRole('textbox', { name: 'Search models' }), { target: { value: 'gemini-2.5' } })

    // A query means "show me every match": the section folds and the match
    // paints in its provider's place. Still exactly once, still starred.
    await vi.waitFor(() => {
      expect(screen.queryByText('Favorites')).toBeNull()
    })

    expect(screen.getAllByText(/Gemini 2\.5/i)).toHaveLength(1)
    expect(screen.getByRole('button', { name: 'Remove from favorites' }).getAttribute('aria-pressed')).toBe('true')
  })

  // Starring is never a pick: the star, a shift-click on the row (the
  // sidebar's pin gesture) and Shift+Enter all toggle in place, the menu stays
  // open, and the same gesture on the moved row undoes it.
  it('the star, shift-click and Shift+Enter toggle a favorite without selecting or closing', async () => {
    const select = renderMenu()
    const key = favoriteModelKey('google', 'gemini-2.5-flash')
    const row = () => screen.getByText('Gemini 2.5').closest('[role="menuitem"]')!
    const star = () => row().querySelector('button[aria-pressed]')!

    await screen.findByText('Gemini 2.5')
    fireEvent.click(star())
    expect($favoriteModels.get()).toEqual([key])
    await screen.findByText('Favorites')

    fireEvent.click(row(), { shiftKey: true })
    expect($favoriteModels.get()).toEqual([])

    const search = screen.getByRole('textbox', { name: 'Search models' })

    fireEvent.change(search, { target: { value: 'gemini-2.5' } })
    await vi.waitFor(() => expect(screen.getAllByText(/Gemini 2\.5/i)).toHaveLength(1))
    fireEvent.keyDown(search, { key: 'Enter', shiftKey: true })
    expect($favoriteModels.get()).toEqual([key])

    expect(select).not.toHaveBeenCalled()
    expect(closeMenu).not.toHaveBeenCalled()
  })
})

describe('the current model the catalog omits (#57534)', () => {
  it('renders the current model as a selectable row when the provider catalog lacks it', async () => {
    getGlobalModelOptions.mockResolvedValue({
      providers: [{ models: ['openai/gpt-4o-mini'], name: 'OpenRouter', slug: 'openrouter' }]
    })

    renderMenu({ model: 'qwen/qwen-2.5-72b-instruct', provider: 'openrouter' })

    // The current model paints as its own family row (humanized name), ahead
    // of the catalog's families, and stays selectable.
    const row = (await screen.findByText('Qwen 2.5 72B Instruct')).closest('[role="menuitem"]')!

    expect(row.textContent).toContain('Qwen 2.5 72B Instruct')
  })

  it('does not inject the current model into a provider that is not the current one', async () => {
    // current.provider has no row; the model must not leak into another group.
    getGlobalModelOptions.mockResolvedValue({
      providers: [{ models: ['openai/gpt-4o-mini'], name: 'OpenRouter', slug: 'openrouter' }]
    })

    renderMenu({ model: 'qwen/qwen-2.5-72b-instruct', provider: 'nous' })

    await screen.findByText('GPT-4o-mini')
    expect(screen.queryByText('Qwen 2.5 72B Instruct')).toBeNull()
  })
})

describe('in-flight local downloads', () => {
  const DOWNLOAD_JOB: LocalRuntimeJob = {
    job_id: 'dl1',
    kind: 'model-download',
    target: 'Qwen3.8 Flash Next (UD-Q4_K_XL)',
    model_id: 'qwen3.8-flash-next',
    status: 'running',
    phase: 'downloading',
    detail: '',
    total_bytes: 100,
    done_bytes: 41,
    percent: 41,
    error: null
  }

  it('shows a downloading model as a disabled progress row in its own Local group', async () => {
    // No llamacpp provider in the catalog (first-ever download).
    queryClient.setQueryData(localModelsKey(localModelsOwner(), 'jobs'), [DOWNLOAD_JOB])
    renderMenu()
    await screen.findByText(/Gemini 3\.1 Pro/i)

    const row = screen.getByText('Qwen3.8 Flash Next (UD-Q4_K_XL)')

    expect(row).toBeTruthy()
    expect(screen.getByText('41%')).toBeTruthy()
    expect(row.closest('[role="menuitem"]')?.getAttribute('aria-disabled')).toBe('true')
  })

  it('shows the download inside the Local provider group when it exists', async () => {
    getGlobalModelOptions.mockResolvedValue({
      providers: [
        { models: ['Qwen3.6-27B-UD-Q4_K_XL'], name: 'Local', slug: 'llamacpp' },
        { models: ['gemini-3.1-pro'], name: 'Google', slug: 'google' }
      ]
    })
    queryClient.setQueryData(localModelsKey(localModelsOwner(), 'jobs'), [DOWNLOAD_JOB])
    renderMenu()

    await screen.findByText(/Qwen3\.6 27B/i)
    expect(screen.getByText('Qwen3.8 Flash Next (UD-Q4_K_XL)')).toBeTruthy()
    // One Local heading — the trailing fallback group must not double up.
    expect(screen.getAllByText('Local').length).toBe(1)
  })

  it('drops the placeholder row once the download settles', async () => {
    queryClient.setQueryData(localModelsKey(localModelsOwner(), 'jobs'), [DOWNLOAD_JOB])
    renderMenu()
    await screen.findByText('Qwen3.8 Flash Next (UD-Q4_K_XL)')

    queryClient.setQueryData(localModelsKey(localModelsOwner(), 'jobs'), [
      { ...DOWNLOAD_JOB, status: 'done', phase: 'done' }
    ])
    await waitFor(() => {
      expect(screen.queryByText('Qwen3.8 Flash Next (UD-Q4_K_XL)')).toBeNull()
    })
  })

  it('hides the local provider group and download rows without the --local flag (strict)', async () => {
    $localModelsEnabled.set(false)
    getGlobalModelOptions.mockResolvedValue({
      providers: [
        { models: ['Qwen3.6-27B-UD-Q4_K_XL'], name: 'Local', slug: 'llamacpp' },
        { models: ['gemini-3.1-pro'], name: 'Google', slug: 'google' }
      ]
    })
    queryClient.setQueryData(localModelsKey(localModelsOwner(), 'jobs'), [DOWNLOAD_JOB])
    renderMenu()

    // Staged models exist and a download is running — none of it shows.
    await screen.findByText(/Gemini 3\.1 Pro/i)
    expect(screen.queryByText(/Qwen3\.6 27B/i)).toBeNull()
    expect(screen.queryByText('Qwen3.8 Flash Next (UD-Q4_K_XL)')).toBeNull()
    expect(screen.queryByText('Local')).toBeNull()
  })
})

// A row shows its model's effort ("Gemini 3.1 Pro  Max"), which reads as a
// fixed model+effort combo unless the row also advertises that the effort is
// editable behind it. The caret is that advertisement, and ArrowRight is the
// way in for anyone not driving the menu with a mouse (#86966).
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

    // The row name no longer carries the variant (`-flash` is its own chip,
    // #118083), so target the truncating name span and walk up to the sub
    // trigger from there.
    const hovered = screen.getByText('Gemini 2.5').closest('[data-slot="dropdown-menu-sub-trigger"]')

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

describe('the catalog renders per-model pricing', () => {
  beforeEach(() => setShowModelPricing(true))
  afterEach(() => setShowModelPricing(false))

  it('keeps prices out of the menu until the pricing setting is on', async () => {
    setShowModelPricing(false)
    getGlobalModelOptions.mockResolvedValue({
      providers: [
        {
          models: ['anthropic/claude-sonnet-5', 'nous/hermes-4'],
          name: 'Nous Portal',
          slug: 'nous',
          pricing: {
            'anthropic/claude-sonnet-5': { input: '$1.60', output: '$8.00', cache: '$0.16', free: false },
            'nous/hermes-4': { input: 'free', output: 'free', cache: null, free: true }
          }
        }
      ]
    })

    renderMenu()

    await screen.findByText('Sonnet 5')
    expect(screen.queryByText('$1.60/$8.00')).toBeNull()
    expect(screen.queryByText('free')).toBeNull()

    act(() => setShowModelPricing(true))
    await screen.findByText('$1.60/$8.00')
    expect(screen.getByText('free')).not.toBeNull()
  })

  it('shows $/Mtok input/output and the sale tag when the provider ships pricing', async () => {
    getGlobalModelOptions.mockResolvedValue({
      providers: [
        {
          models: ['anthropic/claude-sonnet-5'],
          name: 'Nous Portal',
          slug: 'nous',
          pricing: {
            'anthropic/claude-sonnet-5': {
              input: '$1.60',
              output: '$8.00',
              cache: '$0.16',
              free: false,
              discount_percent: 20
            }
          }
        }
      ]
    })

    renderMenu()

    await screen.findByText('$1.60/$8.00')
    expect(screen.queryByText('−20%')).not.toBeNull()
  })

  it('renders a free label for free-tier models', async () => {
    getGlobalModelOptions.mockResolvedValue({
      providers: [
        {
          models: ['nous/hermes-4'],
          name: 'Nous Portal',
          slug: 'nous',
          pricing: {
            'nous/hermes-4': { input: 'free', output: 'free', cache: null, free: true }
          }
        }
      ]
    })

    renderMenu()

    await screen.findByText('free')
  })

  it('appends the cached-read rate next to the uncached input/output prices', async () => {
    getGlobalModelOptions.mockResolvedValue({
      providers: [
        {
          models: ['anthropic/claude-sonnet-5'],
          name: 'Nous Portal',
          slug: 'nous',
          pricing: {
            'anthropic/claude-sonnet-5': {
              input: '$1.60',
              output: '$8.00',
              cache: '$0.16',
              free: false
            }
          }
        }
      ]
    })

    renderMenu()

    const prices = await screen.findByText('$1.60/$8.00')
    expect(prices.parentElement?.textContent).toContain('·$0.16')
    // The cached rate carries its own hover label naming it.
    expect(screen.getByTitle('cached read $0.16/Mtok')).not.toBeNull()
  })

  it('prices a collapsed -fast family from the fast sibling when the base id is unpriced', async () => {
    getGlobalModelOptions.mockResolvedValue({
      providers: [
        {
          models: ['anthropic/claude-sonnet-5', 'anthropic/claude-sonnet-5-fast'],
          name: 'Nous Portal',
          slug: 'nous',
          pricing: {
            'anthropic/claude-sonnet-5-fast': {
              input: '$0.80',
              output: '$4.00',
              cache: null,
              free: false
            }
          }
        }
      ]
    })

    renderMenu()

    await screen.findByText('$0.80/$4.00')
  })

  it('renders no price span when the provider carries no pricing for the model', async () => {
    getGlobalModelOptions.mockResolvedValue({
      providers: [
        {
          models: ['anthropic/claude-sonnet-5'],
          name: 'Nous Portal',
          slug: 'nous'
        }
      ]
    })

    renderMenu()

    // modelDisplayParts prettifies to the bare family name (vendor lives on
    // the provider group row), so the row reads "Sonnet 5", not "Claude
    // Sonnet 5".
    await screen.findByText('Sonnet 5')
    expect(screen.queryByText(/\/Mtok/)).toBeNull()
    expect(screen.queryByText('free')).toBeNull()
  })
})
