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
import { queryClient } from '@/lib/query-client'
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
  $visibleModels.set(null)
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

// models.dev publishes exactly five output modalities; the filter buttons and
// the groupModels narrowing must speak that vocabulary (never dead buttons)
// and read the backend's per-model output_modalities through the ONE call
// site that renders the catalog.
describe('the capability filter (#129106)', () => {
  const CATALOG = {
    providers: [
      {
        models: [
          'nous/hermes-5', // text
          'openai/gpt-image-5', // image
          'openai/whisper-large', // audio output (transcription model)
          'z-ai/glm-6' // no metadata → Unknown
        ],
        name: 'Nous Portal',
        output_modalities: {
          'nous/hermes-5': ['text'],
          'openai/gpt-image-5': ['image'],
          'openai/whisper-large': ['audio'],
          'z-ai/glm-6': []
        },
        slug: 'nous'
      }
    ]
  }

  it('offers only the modalities models.dev actually publishes (plus Unknown)', async () => {
    getGlobalModelOptions.mockResolvedValue(CATALOG)
    renderMenu()

    await screen.findByText('Hermes 5')

    for (const label of ['Text', 'Image', 'Audio', 'Video', 'Unknown']) {
      expect(screen.getByRole('button', { name: label })).not.toBeNull()
    }
    // The reviewer's dead buttons: models.dev has no such output modality,
    // so these can never match a row.

    for (const label of ['Speech', 'Transcription', 'Embeddings']) {
      expect(screen.queryByRole('button', { name: label })).toBeNull()
    }
  })

  it('narrows the catalog to rows whose metadata carries the chosen output modality', async () => {
    getGlobalModelOptions.mockResolvedValue(CATALOG)
    renderMenu()
    await screen.findByText('Hermes 5')

    fireEvent.click(screen.getByRole('button', { name: 'Image' }))
    await vi.waitFor(() => {
      expect(screen.getByText('GPT-image-5')).not.toBeNull()
      expect(screen.queryByText('Hermes 5')).toBeNull()
      expect(screen.queryByText('Whisper Large')).toBeNull()
    })
  })

  it('routes transcription-class models to Audio, their real output modality', async () => {
    getGlobalModelOptions.mockResolvedValue(CATALOG)
    renderMenu()
    await screen.findByText('Hermes 5')

    fireEvent.click(screen.getByRole('button', { name: 'Audio' }))
    await vi.waitFor(() => {
      expect(screen.getByText('Whisper Large')).not.toBeNull()
      expect(screen.queryByText('Hermes 5')).toBeNull()
    })
  })

  it('keeps metadata-less models selectable under Unknown', async () => {
    getGlobalModelOptions.mockResolvedValue(CATALOG)
    renderMenu()
    await screen.findByText('Hermes 5')

    fireEvent.click(screen.getByRole('button', { name: 'Unknown' }))
    await vi.waitFor(() => {
      expect(screen.getByText('GLM 6')).not.toBeNull()
      expect(screen.queryByText('Hermes 5')).toBeNull()
    })
  })

  it('treats a collapsed -fast family as its base model for matching', async () => {
    getGlobalModelOptions.mockResolvedValue({
      providers: [
        {
          models: ['nous/hermes-5', 'nous/hermes-5-fast'],
          name: 'Nous Portal',
          output_modalities: { 'nous/hermes-5': ['text'] },
          slug: 'nous'
        }
      ]
    })
    renderMenu()
    await screen.findByText('Hermes 5')

    fireEvent.click(screen.getByRole('button', { name: 'Text' }))
    await vi.waitFor(() => {
      // The family row (fronting both ids) survives the text filter…
      expect(screen.getByText('Hermes 5')).not.toBeNull()
    })

    fireEvent.click(screen.getByRole('button', { name: 'Image' }))
    await vi.waitFor(() => {
      // …and a modality it doesn't have removes it.
      expect(screen.queryByText('Hermes 5')).toBeNull()
    })
  })

  it('clicking the active filter again clears it and restores every row', async () => {
    getGlobalModelOptions.mockResolvedValue(CATALOG)
    renderMenu()
    await screen.findByText('Hermes 5')

    fireEvent.click(screen.getByRole('button', { name: 'Image' }))
    await vi.waitFor(() => {
      expect(screen.queryByText('Hermes 5')).toBeNull()
    })

    fireEvent.click(screen.getByRole('button', { name: 'Image' }))
    await vi.waitFor(() => {
      expect(screen.getByText('Hermes 5')).not.toBeNull()
      expect(screen.getByText('Whisper Large')).not.toBeNull()
    })
  })
})
