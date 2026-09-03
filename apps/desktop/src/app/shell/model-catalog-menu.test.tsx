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

  it('renders the configured primary then fallbacks instead of alphabetical providers', async () => {
    setVisibleModels(
      new Set([
        modelVisibilityKey('openai-codex', 'gpt-5.6-sol'),
        modelVisibilityKey('anthropic', 'claude-opus-5'),
        modelVisibilityKey('alibaba', 'qwen3.8-max'),
        modelVisibilityKey('alibaba', 'deepseek-v4-flash-0731'),
        modelVisibilityKey('alibaba', 'deepseek-v4-pro')
      ])
    )
    getGlobalModelOptions.mockResolvedValue({
      preferred_models: [
        { provider: 'openai-codex', model: 'gpt-5.6-sol' },
        { provider: 'anthropic', model: 'claude-opus-5' },
        { provider: 'alibaba', model: 'qwen3.8-max' },
        { provider: 'alibaba', model: 'deepseek-v4-flash-0731' },
        { provider: 'alibaba', model: 'deepseek-v4-pro' }
      ],
      providers: [
        { models: ['deepseek-v4-pro', 'qwen3.8-max', 'deepseek-v4-flash-0731'], name: 'Alibaba', slug: 'alibaba' },
        { models: ['claude-opus-5'], name: 'Anthropic', slug: 'anthropic' },
        { models: ['gpt-5.6-sol'], name: 'ChatGPT', slug: 'openai-codex' }
      ]
    })

    renderMenu()
    await screen.findByText(/GPT-5\.6-sol/i)

    const labels = screen.getAllByRole('menuitem').map(row => row.textContent ?? '')
    const indexOf = (needle: RegExp) => labels.findIndex(label => needle.test(label))

    expect(indexOf(/GPT-5\.6-sol/i)).toBeLessThan(indexOf(/Opus 5/i))
    expect(indexOf(/Qwen3\.8 Max/i)).toBeLessThan(indexOf(/Deepseek V4 Flash/i))
    expect(indexOf(/Deepseek V4 Flash/i)).toBeLessThan(indexOf(/Deepseek V4 Pro/i))
  })

  it('uses the canonical xai-oauth identity emitted for the grok-oauth config alias', async () => {
    setVisibleModels(
      new Set([modelVisibilityKey('xai-oauth', 'grok-4.6'), modelVisibilityKey('anthropic', 'claude-opus-5')])
    )
    getGlobalModelOptions.mockResolvedValue({
      preferred_models: [
        { provider: 'xai-oauth', model: 'grok-4.6' },
        { provider: 'anthropic', model: 'claude-opus-5' }
      ],
      providers: [
        { models: ['claude-opus-5'], name: 'Anthropic', slug: 'anthropic' },
        { models: ['grok-4.6'], name: 'xAI', slug: 'xai-oauth' }
      ]
    })

    renderMenu()
    await screen.findByText(/Grok 4\.6/i)

    const labels = screen.getAllByRole('menuitem').map(row => row.textContent ?? '')
    const indexOf = (needle: RegExp) => labels.findIndex(label => needle.test(label))

    expect(indexOf(/Grok 4\.6/i)).toBeLessThan(indexOf(/Opus 5/i))
  })

  it('ranks a collapsed family by its preferred fast-model member', async () => {
    setVisibleModels(
      new Set([modelVisibilityKey('alibaba', 'qwen3.8-max'), modelVisibilityKey('alibaba', 'deepseek-v4-pro')])
    )
    getGlobalModelOptions.mockResolvedValue({
      preferred_models: [
        { provider: 'alibaba', model: 'qwen3.8-max-fast' },
        { provider: 'alibaba', model: 'deepseek-v4-pro' }
      ],
      providers: [
        {
          models: ['qwen3.8-max-fast', 'deepseek-v4-pro', 'qwen3.8-max'],
          name: 'Alibaba',
          slug: 'alibaba'
        }
      ]
    })

    renderMenu()
    await screen.findByText(/Qwen3\.8 Max/i)

    const labels = screen.getAllByRole('menuitem').map(row => row.textContent ?? '')
    const indexOf = (needle: RegExp) => labels.findIndex(label => needle.test(label))

    expect(indexOf(/Qwen3\.8 Max/i)).toBeLessThan(indexOf(/Deepseek V4 Pro/i))
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
