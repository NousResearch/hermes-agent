import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import { DropdownMenu, DropdownMenuContent } from '@/components/ui/dropdown-menu'
import { $localModelsEnabled } from '@/store/local-models-flag'
import { $localRuntimeJobs } from '@/store/local-runtime-jobs'
import { $pinnedModelKeys, pinModel } from '@/store/model-pinned'
import {
  $modelVisibilityOpen,
  $visibleModels,
  modelVisibilityKey,
  setModelVisibilityOpen,
  setVisibleModels
} from '@/store/model-visibility'
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
    const { $localRuntimeJobs } = await import('@/store/local-runtime-jobs')

    return { jobs: [...$localRuntimeJobs.get()] }
  }),
  getLocalModelsStatus: vi.fn().mockResolvedValue({ loading: {} }),
  setApiRequestProfile: vi.fn()
}))

beforeEach(() => {
  $visibleModels.set(null)
  $localRuntimeJobs.set([])
  $pinnedModelKeys.set([])
  // These suites exercise the local-models rows, which ship behind --local.
  $localModelsEnabled.set(true)
  setModelVisibilityOpen(false)
  getGlobalModelOptions.mockResolvedValue({
    providers: [{ models: ['gemini-3.1-pro', 'gemini-2.5-flash'], name: 'Google', slug: 'google' }]
  })
})

afterEach(() => {
  cleanup()
  // The backend mock echoes this snapshot; retire fixture jobs before jsdom
  // disappears so an in-flight app-level poll cannot schedule another tick.
  $localRuntimeJobs.set([])
  $pinnedModelKeys.set([])
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
      // The fold makes this id-style query highlight the spaced label: the
      // row renders as <mark>Gemini 3.1</mark> + ' Pro'.
      expect(screen.getByText('Gemini 3.1', { selector: 'mark' })).toBeDefined()
      // Display name is "Gemini 3.1 pro" (no title-case for gemini ids); the
      // row label span carries it (plus the effort meta suffix).
      expect(
        screen.getByText((_, element) =>
          Boolean(element?.classList.contains('truncate') && (element?.textContent ?? '').startsWith('Gemini 3.1 pro'))
        )
      ).toBeDefined()
    })
  })

  it('offers Edit Models without the host wiring it up', async () => {
    renderMenu()
    await screen.findByText(/Gemini 3\.1 Pro/i)

    fireEvent.click(screen.getByText('Edit models…'))

    expect($modelVisibilityOpen.get()).toBe(true)
  })
})

describe('pinned models', () => {
  it('renders a Pinned section above the provider groups, and does not duplicate the row', async () => {
    pinModel(modelVisibilityKey('google', 'gemini-3.1-pro'))

    renderMenu()

    await screen.findByText('Pinned')
    // Exactly one "Gemini 3.1 Pro" row — pinned, not also under Google.
    expect(screen.getAllByText(/Gemini 3\.1 Pro/i).length).toBe(1)
  })

  it('shows no Pinned section when nothing is pinned', async () => {
    renderMenu()
    await screen.findByText(/Gemini 3\.1 Pro/i)

    expect(screen.queryByText('Pinned')).toBeNull()
  })

  it('pinning a model outside the Edit Models shortlist still surfaces it via the Pinned section', async () => {
    // Only gemini-2.5-flash is in the curated shortlist; pin the OTHER model.
    setVisibleModels(new Set([modelVisibilityKey('google', 'gemini-2.5-flash')]))
    pinModel(modelVisibilityKey('google', 'gemini-3.1-pro'))

    renderMenu()

    await screen.findByText('Pinned')
    expect(screen.getByText(/Gemini 3\.1 Pro/i)).toBeTruthy()
  })

  it('clicking the pin toggle pins a model without selecting it', async () => {
    const select = renderMenu()
    await screen.findByText(/Gemini 3\.1 Pro/i)

    const row = screen.getByText(/Gemini 3\.1 Pro/i).closest('[role="menuitem"]')
    const pinButton = row?.querySelector('button[aria-label="Pin model"]')

    expect(pinButton).toBeTruthy()
    fireEvent.click(pinButton!)

    expect($pinnedModelKeys.get()).toContain(modelVisibilityKey('google', 'gemini-3.1-pro'))
    // The click must not have committed the model.
    expect(select).not.toHaveBeenCalled()
  })

  it('clicking an already-pinned toggle unpins it', async () => {
    pinModel(modelVisibilityKey('google', 'gemini-3.1-pro'))
    renderMenu()

    await screen.findByText('Pinned')
    const pinButton = screen.getByRole('button', { name: 'Unpin model' })
    fireEvent.click(pinButton)

    expect($pinnedModelKeys.get()).not.toContain(modelVisibilityKey('google', 'gemini-3.1-pro'))
  })

  it('a stale pin (model no longer in the catalog) is silently dropped, no crash', async () => {
    pinModel(modelVisibilityKey('google', 'gemini-retired-model'))

    renderMenu()

    await screen.findByText(/Gemini 3\.1 Pro/i)
    expect(screen.queryByText('Pinned')).toBeNull()
  })

  it('search still narrows the Pinned section like any other row', async () => {
    pinModel(modelVisibilityKey('google', 'gemini-3.1-pro'))
    pinModel(modelVisibilityKey('google', 'gemini-2.5-flash'))

    renderMenu()
    await screen.findByText('Pinned')

    const input = screen.getByRole('textbox', { name: 'Search models' })
    fireEvent.change(input, { target: { value: '2.5' } })

    await waitFor(() => {
      expect(screen.queryByText(/Gemini 3\.1 Pro/i)).toBeNull()
      // The fold makes this id-style query highlight the spaced label: the
      // row renders as <mark>2.5</mark> inside "Gemini 2.5 flash".
      expect(screen.getByText('2.5', { selector: 'mark' })).toBeTruthy()
    })
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
    $localRuntimeJobs.set([DOWNLOAD_JOB])
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
    $localRuntimeJobs.set([DOWNLOAD_JOB])
    renderMenu()

    await screen.findByText(/Qwen3\.6 27B/i)
    expect(screen.getByText('Qwen3.8 Flash Next (UD-Q4_K_XL)')).toBeTruthy()
    // One Local heading — the trailing fallback group must not double up.
    expect(screen.getAllByText('Local').length).toBe(1)
  })

  it('drops the placeholder row once the download settles', async () => {
    $localRuntimeJobs.set([DOWNLOAD_JOB])
    renderMenu()
    await screen.findByText('Qwen3.8 Flash Next (UD-Q4_K_XL)')

    $localRuntimeJobs.set([{ ...DOWNLOAD_JOB, status: 'done', phase: 'done' }])
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
    $localRuntimeJobs.set([DOWNLOAD_JOB])
    renderMenu()

    // Staged models exist and a download is running — none of it shows.
    await screen.findByText(/Gemini 3\.1 Pro/i)
    expect(screen.queryByText(/Qwen3\.6 27B/i)).toBeNull()
    expect(screen.queryByText('Qwen3.8 Flash Next (UD-Q4_K_XL)')).toBeNull()
    expect(screen.queryByText('Local')).toBeNull()
  })
})
