import type { ModelOptionsResult } from '@hermes/shared'
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

// `refreshModelOptions` calls `requestModelOptions` from INSIDE its module —
// vitest does not intercept intra-module calls (probed), so the refresh click
// runs the real request path and bottoms out at this `@/hermes` seam (REST,
// no gateway/request props in these tests). `requestModelOptions` itself
// intercepts only the component's open-time `useQuery` fetch.
const getGlobalModelOptions = vi.fn()

vi.mock('@/hermes', () => ({
  getGlobalModelOptions: (...args: unknown[]) => getGlobalModelOptions(...args),
  getLocalModelsStatus: vi.fn().mockResolvedValue({ loading: {} })
}))
vi.mock('@/lib/model-options', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  requestModelOptions: vi.fn()
}))

import { requestModelOptions } from '@/lib/model-options'

stubResizeObserver()
stubMenuDomApis()

const OPTIONS: ModelOptionsResult = {
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

  // The client is returned so tests can spy on cache operations (e.g. the
  // exact invalidation scope the refresh-failure path touches).
  return { ...render(element), client }
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

describe('ModelPickerDialog undiscovered configured providers', () => {
  // #49656: a provider configured under custom_providers: arrives as a
  // user-defined row with an empty model list until its /v1/models catalog is
  // fetched. The picker must keep it visible with a hint row, not drop it.
  it('keeps a user-defined provider with an empty catalog visible with a hint row', async () => {
    vi.mocked(requestModelOptions).mockResolvedValue({
      providers: [
        {
          slug: 'custom:my-provider',
          name: 'my-provider',
          is_user_defined: true,
          authenticated: true,
          models: []
        }
      ]
    })
    renderPicker()

    expect(await screen.findByText('my-provider')).toBeTruthy()

    const hint = screen.getByText('No models discovered yet')

    // Non-selectable: cmdk marks the item unselectable (no onSelect path).
    expect(hint.closest('[cmdk-item]')?.getAttribute('aria-disabled')).toBe('true')
  })

  it('hides an empty provider the user did not configure', async () => {
    vi.mocked(requestModelOptions).mockResolvedValue({
      providers: [{ slug: 'canonical', name: 'Some Canonical', models: [] }]
    })
    renderPicker()

    // Wait until the payload has rendered: with nothing selectable left,
    // cmdk shows its empty state.
    await screen.findByText('No models found.')

    expect(screen.queryByText('Some Canonical')).toBeNull()
    expect(screen.queryByText('No models discovered yet')).toBeNull()
  })

  it('refreshes the whole catalog from the footer button', async () => {
    renderPicker()
    await screen.findByText('Hermes-4.5')

    fireEvent.click(screen.getByRole('button', { name: 'Refresh models' }))

    // The shared helper calls `requestModelOptions` intra-module (immune to
    // vi.mock), so the refresh:true lands on the REST seam below.
    await waitFor(() => {
      expect(getGlobalModelOptions).toHaveBeenCalledWith(expect.objectContaining({ refresh: true }), 'default')
    })
  })

  it('renders no hint row for providers whose catalog has models', async () => {
    renderPicker()

    expect(await screen.findByText('Hermes-4.5')).toBeTruthy()
    expect(screen.queryByText('No models discovered yet')).toBeNull()
  })

  // Search parity with the composer menu (model-catalog-menu's groupModels
  // skips the undiscovered group while a query is active): a query means
  // "show me matches", so a provider with nothing to match must vanish
  // entirely — heading and hint row alike.
  it('hides the undiscovered provider and its hint row while searching', async () => {
    vi.mocked(requestModelOptions).mockResolvedValue({
      providers: [
        {
          slug: 'custom:my-provider',
          name: 'my-provider',
          is_user_defined: true,
          authenticated: true,
          models: []
        },
        { slug: 'nous', name: 'Nous', models: ['Hermes-4.5'], authenticated: true }
      ]
    })
    renderPicker()

    expect(await screen.findByText('No models discovered yet')).toBeTruthy()

    fireEvent.change(screen.getByRole('combobox'), { target: { value: 'Hermes' } })

    await waitFor(() => {
      expect(screen.queryByText('No models discovered yet')).toBeNull()
      expect(screen.queryByText('my-provider')).toBeNull()
    })
    // The normal provider's matching rows are untouched by the fix (row text
    // is split by HighlightMatches spans — compare via textContent).
    expect(screen.getAllByRole('option').some(row => row.textContent?.includes('Hermes-4.5'))).toBe(true)
  })

  // A failed refresh must re-enable the button (the refreshing flag clears)
  // and re-fetch ONLY this picker's scope — invalidating the root
  // ['model-options'] key would refetch every model-options query in the
  // window, including other tiles' owner-scoped catalogs. The failure is
  // driven through the REST seam (see the mock note above): the helper's
  // intra-module requestModelOptions call is immune to vi.mock.
  it('re-enables refresh and invalidates only this scope when the refresh fails', async () => {
    const { client } = renderPicker()
    await screen.findByText('Hermes-4.5')

    const invalidateSpy = vi.spyOn(client, 'invalidateQueries')
    // The cancel is the point of the shared helper: an in-flight fetch for the
    // same key must not be able to land after the refreshed catalog and
    // overwrite it. `revert: false` is the second argument, not a filter field.
    const cancelSpy = vi.spyOn(client, 'cancelQueries')

    // The open-time fetch (mocked requestModelOptions) already resolved; only
    // the refresh click's REST call fails now.
    getGlobalModelOptions.mockRejectedValueOnce(new Error('network down'))

    const refreshButton = screen.getByRole('button', { name: 'Refresh models' })

    fireEvent.click(refreshButton)

    await waitFor(() => {
      expect(invalidateSpy).toHaveBeenCalled()
    })
    expect(cancelSpy).toHaveBeenCalledWith(
      { queryKey: ['model-options', 'default', 'global'] },
      { revert: false }
    )
    expect(refreshButton.hasAttribute('disabled')).toBe(false)
    expect(invalidateSpy).toHaveBeenCalledWith({
      queryKey: ['model-options', 'default', 'global']
    })
  })
})
