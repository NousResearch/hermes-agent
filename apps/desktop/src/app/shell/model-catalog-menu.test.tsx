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
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import { DropdownMenu, DropdownMenuContent } from '@/components/ui/dropdown-menu'
import { queryClient } from '@/lib/query-client'
import { $localModelsEnabled } from '@/store/local-models-flag'
import { localModelsKey, localModelsOwner } from '@/store/local-runtime-jobs'
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

    const row = (await screen.findByText(/Gemini 2\.5 Flash/i)).closest('[role="menuitem"]')!

    expect(row.textContent).not.toContain('Ultra')
    cleanup()

    renderMenu({ model: 'gemini-2.5-flash', provider: 'google' })

    const settled = (await screen.findByText(/Gemini 2\.5 Flash/i)).closest('[role="menuitem"]')!

    expect(settled.textContent).toContain('Ultra')
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

// Distinct ids that prettify to the SAME label are the whole point of showing
// the id: without it the two rows are indistinguishable.
describe('a row shows the raw id it maps to', () => {
  it('keeps two ids that share a display name apart', async () => {
    getGlobalModelOptions.mockResolvedValue({
      providers: [{ models: ['b-ai/glm-5.3-flash', 'kios-ai/glm-5.3-flash'], name: 'Router', slug: 'router' }]
    })

    renderMenu()

    await screen.findAllByText('Glm 5.3 Flash')

    const rowFor = (id: string) => screen.getByText(id).closest('[role="menuitem"]')
    const baseRow = rowFor('b-ai/glm-5.3-flash')
    const kiosRow = rowFor('kios-ai/glm-5.3-flash')

    expect(baseRow).not.toBeNull()
    expect(kiosRow).not.toBeNull()
    expect(baseRow?.textContent).toContain('Glm 5.3 Flash')
    expect(kiosRow?.textContent).toContain('Glm 5.3 Flash')
    expect(kiosRow).not.toBe(baseRow)

    // The filter matches the id too, so an id-only query marks the id line.
    fireEvent.change(screen.getByRole('textbox', { name: 'Search models' }), { target: { value: 'kios' } })

    await vi.waitFor(() => {
      expect(screen.getByText('kios', { selector: 'mark' })).toBeDefined()
      expect(screen.queryByText('b-ai/glm-5.3-flash')).toBeNull()
    })
  })
})
