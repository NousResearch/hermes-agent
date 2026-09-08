import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, render, screen, waitFor } from '@testing-library/react'
import type { ReactElement } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'
import { $localModelsEnabled } from '@/store/local-models-flag'
import { $localRuntimeJobs } from '@/store/local-runtime-jobs'
import { stubMenuDomApis, stubResizeObserver } from '@/test/jsdom'
import type { LocalRuntimeJob, ModelOptionsResponse } from '@/types/hermes'

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
