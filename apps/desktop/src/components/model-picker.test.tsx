import type { ModelOptionsResponse } from '@hermes/shared'
import { fuzzyRank, modelSearchText } from '@hermes/shared'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import type { ReactElement } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'
import { $localModelsEnabled } from '@/store/local-models-flag'
import { $localRuntimeJobs } from '@/store/local-runtime-jobs'
import { stubMenuDomApis, stubResizeObserver } from '@/test/jsdom'
import type { LocalRuntimeJob } from '@/types/hermes'

import { ModelPickerDialog } from './model-picker'

vi.mock('@/hermes', () => ({
  getLocalModelsStatus: vi.fn().mockResolvedValue({ loading: {} })
}))
vi.mock('@/lib/model-options', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  requestModelOptions: vi.fn()
}))

import { requestModelOptions } from '@/lib/model-options'

stubResizeObserver()
stubMenuDomApis()

const OPTIONS: ModelOptionsResponse = {
  model: 'Qwen3.6-27B-UD-Q4_K_XL',
  provider: 'llamacpp',
  providers: [
    {
      slug: 'llamacpp',
      name: 'Local',
      models: ['Qwen3.6-27B-UD-Q4_K_XL'],
      is_current: true,
      authenticated: true
    },
    {
      slug: 'nous',
      name: 'Nous',
      models: ['Hermes-4.5'],
      authenticated: true
    }
  ]
}

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

function renderPicker(ui?: Partial<Parameters<typeof ModelPickerDialog>[0]>) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })

  const element: ReactElement = (
    <QueryClientProvider client={client}>
      <I18nProvider>
        <ModelPickerDialog
          currentModel="Qwen3.6-27B-UD-Q4_K_XL"
          currentProvider="llamacpp"
          onOpenChange={() => undefined}
          onSelect={() => undefined}
          open
          {...ui}
        />
      </I18nProvider>
    </QueryClientProvider>
  )

  return render(element)
}

beforeEach(() => {
  vi.mocked(requestModelOptions).mockResolvedValue(OPTIONS)
  $localRuntimeJobs.set([])
  // These suites exercise the local-models rows, which ship behind --local.
  $localModelsEnabled.set(true)
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

describe('recorded credential limits', () => {
  it.each(['unknown', 'free', 'paid'] as const)(
    'preserves independent Nous %s entitlement despite a pool warning',
    async tier => {
      const warning =
        'Credential pool has a recorded limit; remote availability not checked. Recorded reset: 2100-01-01.'
      const unavailable = tier === 'unknown' ? ['free-model', 'paid-model'] : tier === 'free' ? ['paid-model'] : []
      const provider = {
        slug: 'nous',
        name: 'Nous',
        models: ['free-model', 'paid-model'],
        authenticated: true,
        warning,
        unavailable_models: unavailable,
        free_tier: tier === 'free',
        free_tier_pending: tier === 'unknown'
      }
      vi.mocked(requestModelOptions).mockResolvedValue({ providers: [provider] })
      const onSelect = vi.fn()
      renderPicker({ onSelect })
      await screen.findByText('paid-model')
      expect(screen.getByText(warning)).toBeTruthy()
      for (const model of ['free-model', 'paid-model']) {
        const item = screen.getByText(model).closest('[cmdk-item]')!
        expect(item.getAttribute('aria-disabled') === 'true').toBe(unavailable.includes(model))
        if (unavailable.includes(model)) fireEvent.click(item)
      }
      expect(onSelect).not.toHaveBeenCalled()
    }
  )

  it('keeps the catalog visible with its recorded-state warning, not a login prompt', async () => {
    const warning =
      'Credential pool has a recorded limit; remote availability and model scope not checked. Recorded reset: 2100-01-01T00:00:00+00:00.'
    vi.mocked(requestModelOptions).mockResolvedValue({
      providers: [
        { slug: 'openai-codex', name: 'OpenAI Codex', models: ['gpt-5.3-codex'], authenticated: true, warning }
      ]
    })
    renderPicker()
    expect(await screen.findByText('gpt-5.3-codex')).toBeTruthy()
    expect(screen.getByText(warning)).toBeTruthy()
    expect(screen.getByText('gpt-5.3-codex').closest('[cmdk-item]')?.getAttribute('aria-disabled')).not.toBe('true')
  })
})

describe('ModelPickerDialog download rows', () => {
  it('shows an in-flight download as a disabled progress row in the Local group', async () => {
    $localRuntimeJobs.set([DOWNLOAD_JOB])
    renderPicker()

    expect(await screen.findByText('Qwen3.6-27B-UD-Q4_K_XL')).toBeTruthy()

    const row = screen.getByText('Qwen3.8 Flash Next (UD-Q4_K_XL)')

    expect(row).toBeTruthy()
    expect(screen.getByText('41%')).toBeTruthy()

    // Disabled: cmdk marks the item unselectable.
    const item = row.closest('[cmdk-item]')

    expect(item?.getAttribute('aria-disabled')).toBe('true')
  })

  it('shows a first-ever download under its own Local group when no local provider exists yet', async () => {
    $localRuntimeJobs.set([DOWNLOAD_JOB])
    vi.mocked(requestModelOptions).mockResolvedValue({
      providers: [OPTIONS.providers![1]]
    })
    renderPicker()

    expect(await screen.findByText('Hermes-4.5')).toBeTruthy()
    expect(screen.getByText('Qwen3.8 Flash Next (UD-Q4_K_XL)')).toBeTruthy()
    expect(screen.getByText('41%')).toBeTruthy()
  })

  it('quickstart shows while downloading but not during later phases', async () => {
    const quickstart: LocalRuntimeJob = { ...DOWNLOAD_JOB, job_id: 'q1', kind: 'quickstart', phase: 'downloading' }

    $localRuntimeJobs.set([quickstart])
    renderPicker()
    expect(await screen.findByText('Qwen3.8 Flash Next (UD-Q4_K_XL)')).toBeTruthy()

    // The model is staged once quickstart moves on to activating it — the
    // placeholder row must leave rather than sit beside the real model.
    $localRuntimeJobs.set([{ ...quickstart, phase: 'starting-server' }])
    await waitFor(() => {
      expect(screen.queryByText('Qwen3.8 Flash Next (UD-Q4_K_XL)')).toBeNull()
    })
  })

  it('refetches the model options when a download it saw running completes', async () => {
    $localRuntimeJobs.set([DOWNLOAD_JOB])
    renderPicker()
    await screen.findByText('Qwen3.6-27B-UD-Q4_K_XL')

    expect(vi.mocked(requestModelOptions).mock.calls.length).toBe(1)

    $localRuntimeJobs.set([{ ...DOWNLOAD_JOB, status: 'done', phase: 'done' }])
    await waitFor(() => {
      expect(vi.mocked(requestModelOptions).mock.calls.length).toBe(2)
    })
  })
})

describe('ModelPickerDialog search ranking', () => {
  // Rows must come out in the order the shared fuzzyRank produces — the same
  // helper the web and TUI pickers use — so a query ranks identically on
  // every surface. Curated order puts the scattered match first; the ranked
  // order does not, which is what proves the picker is not substring-filtering.
  const MODELS = ['glm-4.6-omni', 'claude-sonnet-4', 'gpt-4o']

  it('orders model rows exactly as the shared fuzzyRank does', async () => {
    vi.mocked(requestModelOptions).mockResolvedValue({
      providers: [{ slug: 'nous', name: 'Nous', models: MODELS, authenticated: true }]
    })
    renderPicker({ currentModel: 'gpt-4o', currentProvider: 'nous' })
    await screen.findByText('gpt-4o')

    const query = 'g4o'
    fireEvent.change(screen.getByRole('combobox'), { target: { value: query } })

    const expected = fuzzyRank(MODELS, query, modelSearchText).map(r => r.item)

    expect(expected).not.toEqual(MODELS.filter(m => expected.includes(m)))
    await waitFor(() => {
      const rows = screen.getAllByRole('option').map(el => el.textContent?.trim())

      expect(rows).toEqual(expected)
    })
  })

  // Regression guard: main folded `[-_.]` on both sides (foldIncludes); the
  // shared ranker must too, or a query typed with the "wrong" separator
  // drops every row while the highlighter (which still folds) disagrees.
  it.each([
    ['gpt.4o', 'gpt-4o'],
    ['claude_3', 'claude-3-opus'],
    ['qwen3-8', 'qwen3.8-flash']
  ])('separator variant %s still lists %s', async (query, expected) => {
    const catalog = ['gpt-4o', 'claude-3-opus', 'qwen3.8-flash']

    vi.mocked(requestModelOptions).mockResolvedValue({
      providers: [{ slug: 'nous', name: 'Nous', models: catalog, authenticated: true }]
    })
    renderPicker({ currentModel: 'gpt-4o', currentProvider: 'nous' })
    await screen.findByText('gpt-4o')

    fireEvent.change(screen.getByRole('combobox'), { target: { value: query } })

    await waitFor(() => {
      const rows = screen.getAllByRole('option').map(el => el.textContent?.trim())

      expect(rows).toContain(expected)
    })
  })
})
