import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { getAutomationBlueprints, getCronJobOutput, getCronJobOutputs, getCronJobs } from '@/hermes'
import { TRANSLATIONS } from '@/i18n'
import { $cronFocus, setCronJobs } from '@/store/cron'
import { setShowAllProfiles } from '@/store/profile'

import { CronJobRuns, CronView } from './index'

function deferred<T>() {
  let resolve!: (value: T) => void

  const promise = new Promise<T>(res => {
    resolve = res
  })

  return { promise, resolve }
}

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  getAutomationBlueprints: vi.fn(),
  getCronJobOutput: vi.fn(),
  getCronJobOutputs: vi.fn(),
  getCronJobs: vi.fn()
}))

describe('CronJobRuns', () => {
  afterEach(() => {
    cleanup()
    $cronFocus.set(null)
    setShowAllProfiles(false)
    setCronJobs([])
    vi.unstubAllGlobals()
    vi.clearAllMocks()
  })

  it('loads and renders the durable markdown output when a run is clicked', async () => {
    vi.mocked(getCronJobOutputs).mockResolvedValue([
      {
        byte_size: 72,
        created_at: 1_786_435_200,
        filename: '2026-08-11_09-00-00.md',
        id: '2026-08-11_09-00-00'
      }
    ])
    vi.mocked(getCronJobOutput).mockResolvedValue({
      byte_size: 72,
      content: '# Report\n\n| Name | Value |\n| --- | --- |\n| ok | 1 |\n\n[Details](https://example.com)',
      created_at: 1_786_435_200,
      filename: '2026-08-11_09-00-00.md',
      id: '2026-08-11_09-00-00',
      profile: 'worker_alpha'
    })

    render(<CronJobRuns c={TRANSLATIONS.en.cron} jobId="report-job" profile="worker_alpha" />)

    const run = await screen.findByRole('button', { name: /2026-08-11_09-00-00\.md/ })
    fireEvent.click(run)

    expect(getCronJobOutputs).toHaveBeenCalledWith('report-job', 20, 'worker_alpha')
    expect(getCronJobOutput).toHaveBeenCalledWith('report-job', '2026-08-11_09-00-00', 'worker_alpha')
    await waitFor(() => expect(screen.getByRole('heading', { name: 'Report' })).toBeTruthy())
    expect(screen.getByRole('cell', { name: 'ok' })).toBeTruthy()
    expect(screen.getByRole('link', { name: 'Details' }).getAttribute('href')).toBe('https://example.com/')
  })

  it('distinguishes a failed output listing from an empty run history', async () => {
    vi.mocked(getCronJobOutputs).mockRejectedValue(new Error('profile backend unavailable'))

    render(<CronJobRuns c={TRANSLATIONS.en.cron} jobId="report-job" />)

    expect(await screen.findByText(TRANSLATIONS.en.cron.failedLoad)).toBeTruthy()
    expect(screen.queryByText(TRANSLATIONS.en.cron.noRuns)).toBeNull()
  })

  it('reports and consumes a focus target whose retained output is gone', async () => {
    vi.mocked(getCronJobOutputs).mockResolvedValue([])
    $cronFocus.set({
      jobId: 'report-job',
      outputId: '2026-08-10_09-00-00',
      profile: 'worker_alpha'
    })

    render(<CronJobRuns c={TRANSLATIONS.en.cron} jobId="report-job" profile="worker_alpha" />)

    expect((await screen.findByRole('status')).textContent).toContain('2026-08-10_09-00-00.md')
    expect(getCronJobOutput).not.toHaveBeenCalled()
    await waitFor(() => expect($cronFocus.get()).toBeNull())
  })

  it('keeps an output focus target when the run listing fails transiently', async () => {
    const focusTarget = {
      jobId: 'report-job',
      outputId: '2026-08-10_09-00-00',
      profile: 'worker_alpha'
    }

    vi.mocked(getCronJobOutputs).mockRejectedValue(new Error('profile backend unavailable'))
    $cronFocus.set(focusTarget)

    render(<CronJobRuns c={TRANSLATIONS.en.cron} jobId="report-job" profile="worker_alpha" />)

    expect(await screen.findByText(TRANSLATIONS.en.cron.failedLoad)).toBeTruthy()
    expect($cronFocus.get()).toEqual(focusTarget)
    expect(screen.queryByRole('status')).toBeNull()
  })

  it('cannot publish a late output detail after the job profile scope changes', async () => {
    const staleDetail = deferred<Awaited<ReturnType<typeof getCronJobOutput>>>()

    vi.mocked(getCronJobOutputs).mockImplementation(async (_jobId, _limit, profile) => [
      {
        byte_size: 72,
        created_at: profile === 'worker_alpha' ? 1_786_435_200 : 1_786_521_600,
        filename: `${profile}.md`,
        id: `${profile}-output`
      }
    ])
    vi.mocked(getCronJobOutput).mockReturnValue(staleDetail.promise)

    const view = render(<CronJobRuns c={TRANSLATIONS.en.cron} jobId="shared-job" profile="worker_alpha" />)

    fireEvent.click(await screen.findByRole('button', { name: /worker_alpha\.md/ }))
    view.rerender(<CronJobRuns c={TRANSLATIONS.en.cron} jobId="shared-job" profile="worker_beta" />)

    expect(await screen.findByRole('button', { name: /worker_beta\.md/ })).toBeTruthy()

    await act(async () => {
      staleDetail.resolve({
        byte_size: 72,
        content: '# Wrong profile output',
        created_at: 1_786_435_200,
        filename: 'worker_alpha.md',
        id: 'worker_alpha-output',
        profile: 'worker_alpha'
      })
      await staleDetail.promise
    })

    expect(screen.queryByRole('heading', { name: 'Wrong profile output' })).toBeNull()
    expect(screen.queryByText(TRANSLATIONS.en.cron.failedLoad)).toBeNull()
  })

  it('opens a focused duplicate-id job from the owning profile in the full Cron view', async () => {
    const jobs = [
      {
        enabled: true,
        id: 'shared-job',
        name: 'Default report',
        profile: 'default',
        schedule: { expr: '0 9 * * *' }
      },
      {
        enabled: true,
        id: 'shared-job',
        name: 'Worker report',
        profile: 'worker_alpha',
        schedule: { expr: '0 9 * * *' }
      }
    ] as never

    setShowAllProfiles(true)
    setCronJobs(jobs)
    $cronFocus.set({ jobId: 'shared-job', outputId: 'worker-output', profile: 'worker_alpha' })
    vi.mocked(getCronJobs).mockResolvedValue(jobs)
    vi.mocked(getAutomationBlueprints).mockResolvedValue({ blueprints: [] })
    vi.mocked(getCronJobOutputs).mockImplementation(async (_jobId, _limit, profile) =>
      profile === 'worker_alpha'
        ? [{ byte_size: 72, created_at: 1_786_435_200, filename: 'worker-output.md', id: 'worker-output' }]
        : [{ byte_size: 72, created_at: 1_786_435_200, filename: 'default-output.md', id: 'default-output' }]
    )
    vi.mocked(getCronJobOutput).mockResolvedValue({
      byte_size: 72,
      content: '# Worker output',
      created_at: 1_786_435_200,
      filename: 'worker-output.md',
      id: 'worker-output',
      profile: 'worker_alpha'
    })
    vi.stubGlobal('CSS', { escape: (value: string) => value.replace(/["\\]/g, '\\$&') })
    Object.defineProperty(window.HTMLElement.prototype, 'scrollIntoView', {
      configurable: true,
      value: vi.fn()
    })

    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })

    render(
      <QueryClientProvider client={queryClient}>
        <CronView onClose={vi.fn()} />
      </QueryClientProvider>
    )

    await waitFor(() =>
      expect(getCronJobOutput).toHaveBeenCalledWith('shared-job', 'worker-output', 'worker_alpha')
    )
    expect(await screen.findByRole('heading', { name: 'Worker output' })).toBeTruthy()
    expect($cronFocus.get()).toBeNull()
  })
})
