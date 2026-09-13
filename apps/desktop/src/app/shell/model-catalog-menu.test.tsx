import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import { DropdownMenu, DropdownMenuContent } from '@/components/ui/dropdown-menu'
import { $localModelsEnabled } from '@/store/local-models-flag'
import { $localRuntimeJobs } from '@/store/local-runtime-jobs'
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
  vi.clearAllMocks()
})

// A minimal controller — these tests are about the CATALOG's own behaviour
// (what it lists, what it offers), not about what any host does with a pick.
function renderMenu(current: ModelMenuController['current'] = { effort: '', fast: false, model: '', provider: '' }) {
  const select = vi.fn()

  const controller: ModelMenuController = {
    applyPreset: vi.fn(),
    current,
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
          Boolean(
            element?.hasAttribute?.('data-row-label') && (element?.textContent ?? '').startsWith('Gemini 3.1 pro')
          )
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

  it('focuses the search field on open so typing filters immediately', async () => {
    renderMenu()
    await screen.findByText(/Gemini 3\.1 Pro/i)

    await vi.waitFor(() => {
      expect(document.activeElement?.getAttribute('placeholder')).toBe('Search models')
    })
  })
})

describe('multi-upstream rows stay distinguishable', () => {
  const LOCAL_MODELS = ['cmd/deepseek/deepseek-v4-flash', 'cbai/deepseek-v4-flash', 'auto/best-coding']

  function renderLocal(total = 5973, current?: ModelMenuController['current']) {
    getGlobalModelOptions.mockResolvedValue({
      providers: [{ models: LOCAL_MODELS, name: 'Local', slug: 'local', total_models: total }]
    })
    renderMenu(current)
  }

  it('shows the upstream in the id line, no pills', async () => {
    renderLocal()
    await screen.findByText('Best Coding')

    // No qualifier pills — the id line carries the upstream verbatim.
    expect(screen.getByText('cmd/deepseek/deepseek-v4-flash')).toBeDefined()
    expect(screen.getByText('cbai/deepseek-v4-flash')).toBeDefined()
    expect(screen.queryByText('cmd/deepseek')).toBeNull()
  })

  it('meters the effort as text next to the name', async () => {
    renderLocal()
    await screen.findByText('Best Coding')
    // Native tooltip carries the exact id — truncated rows reveal on hover.
    const row = screen.getByTitle('cmd/deepseek/deepseek-v4-flash')

    expect(row).toBeDefined()
    // Medium effort renders as a text suffix on the row.
    expect(row.textContent).toContain('Med')
  })

  it('lists families A–Z by default', async () => {
    renderLocal()
    await screen.findByText('Best Coding')
    const names = [...document.querySelectorAll('[data-row-label]')].map(el => el.textContent)

    // Tie on "Deepseek V4 Flash" breaks by id: cbai before cmd.
    expect(names).toEqual(['Best Coding', 'Deepseek V4 Flash', 'Deepseek V4 Flash'])
  })

  it('badges the provider header with shown-of-total counts', async () => {
    renderLocal()
    await screen.findByText('3 of 5,973')
  })

  it('still pins the active model first while searching', async () => {
    renderLocal(5973, { effort: '', fast: false, model: 'cmd/deepseek/deepseek-v4-flash', provider: 'local' })
    await screen.findByText('Best Coding')

    fireEvent.change(screen.getByRole('textbox', { name: 'Search models' }), { target: { value: 'flash' } })

    await vi.waitFor(() => {
      const rows = [...document.querySelectorAll('[data-row-label]')]

      // Alphabetically cbai would lead — the active cmd pick stays on top.
      expect(rows.map(el => el.textContent)).toEqual(['Deepseek V4 Flash', 'Deepseek V4 Flash'])
      expect(rows[0].closest('span[title]')?.getAttribute('title')).toContain('cmd/deepseek/deepseek-v4-flash')
    })
  })

  it('pins the active model first, rest stays A–Z', async () => {
    renderLocal(5973, { effort: '', fast: false, model: 'cbai/deepseek-v4-flash', provider: 'local' })
    await screen.findByText('Best Coding')
    const names = [...document.querySelectorAll('[data-row-label]')]

    expect(names.map(el => el.textContent)).toEqual(['Deepseek V4 Flash', 'Best Coding', 'Deepseek V4 Flash'])
    expect(names[0].closest('span[title]')?.getAttribute('title')).toContain('cbai/deepseek-v4-flash')
  })

  it('glides a clipped label on hover and snaps back on leave', async () => {
    renderLocal()
    await screen.findByText('Best Coding')
    const marquee = document.querySelector('[data-marquee="name:cmd/deepseek/deepseek-v4-flash"]') as HTMLElement
    const inner = marquee.firstElementChild as HTMLElement

    // jsdom reports zero sizes — stage a 200px overflow.
    Object.defineProperty(inner, 'scrollWidth', { configurable: true, value: 300 })
    Object.defineProperty(marquee, 'clientWidth', { configurable: true, value: 100 })

    // Hover bubbles to the row, which scrolls every clipped line together.
    fireEvent.mouseOver(marquee)
    await vi.waitFor(() => {
      expect(inner.style.transform).toContain('translateX(-200px)')
    })

    fireEvent.mouseOut(marquee)
    expect(inner.style.transform).toBe('')
  })

  it('leaves fitting labels put on hover', async () => {
    renderLocal()
    await screen.findByText('Best Coding')
    const marquee = document.querySelector('[data-marquee="name:auto/best-coding"]') as HTMLElement
    const inner = marquee.firstElementChild as HTMLElement

    fireEvent.mouseOver(marquee)
    expect(inner.style.transform).toBe('')
  })

  it('renders the variant tag instead of dropping it', async () => {
    getGlobalModelOptions.mockResolvedValue({
      providers: [{ models: ['deepseek/deepseek-v4-pro-thinking'], name: 'Local', slug: 'local' }]
    })
    renderMenu()
    await screen.findByText('Deepseek V4 Pro')
    expect(screen.getByText('Thinking')).toBeDefined()
  })
})

describe('token search', () => {
  it('matches unordered tokens across id and upstream', async () => {
    getGlobalModelOptions.mockResolvedValue({
      providers: [
        { models: ['opencode/deepseek-v4-flash', 'cmd/deepseek/deepseek-v4-flash'], name: 'Mixed', slug: 'mixed' }
      ]
    })
    renderMenu()
    await screen.findAllByText('Deepseek V4 Flash')

    fireEvent.change(screen.getByRole('textbox', { name: 'Search models' }), {
      target: { value: 'deepseek v4 flash opencode' }
    })

    await vi.waitFor(() => {
      const names = [...document.querySelectorAll('[data-row-label]')].map(el => el.textContent)

      expect(names).toEqual(['Deepseek V4 Flash'])
    })
    // The survivor is the opencode route.
    expect(screen.getByTitle('opencode/deepseek-v4-flash')).toBeDefined()
  })
})

describe('search render cap', () => {
  const MANY = Array.from({ length: 250 }, (_, i) => `zz-model-${String(i).padStart(3, '0')}`)

  it('renders the first 100 matches with a narrowing note', async () => {
    getGlobalModelOptions.mockResolvedValue({ providers: [{ models: MANY, name: 'Many', slug: 'many' }] })
    renderMenu()
    await screen.findByText('Zz Model 000')

    fireEvent.change(screen.getByRole('textbox', { name: 'Search models' }), { target: { value: 'zz-model' } })

    await vi.waitFor(() => {
      expect(screen.getByText(/keep typing to narrow/i)).toBeDefined()
    })
    expect(document.querySelectorAll('[data-row-label]').length).toBe(100)
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
