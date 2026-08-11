import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { getCronJobOutput, getCronJobOutputs } from '@/hermes'
import { TRANSLATIONS } from '@/i18n'
import { $cronFocus } from '@/store/cron'

import { CronJobRuns } from './index'

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  getCronJobOutput: vi.fn(),
  getCronJobOutputs: vi.fn()
}))

describe('CronJobRuns', () => {
  afterEach(() => {
    cleanup()
    $cronFocus.set(null)
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
})
