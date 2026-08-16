import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { getCronJobOutputs, getCronJobRuns } from '@/hermes'
import { TRANSLATIONS } from '@/i18n'

import { CronJobSidebarRuns } from './cron-jobs-section'

vi.mock('@/components/pane-shell/pane-visibility', () => ({
  usePaneVisible: () => true
}))

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  getCronJobOutputs: vi.fn(),
  getCronJobRuns: vi.fn()
}))

describe('CronJobSidebarRuns', () => {
  afterEach(() => {
    cleanup()
    vi.clearAllMocks()
  })

  it('opens durable output for a no-agent cron run', async () => {
    vi.mocked(getCronJobOutputs).mockResolvedValue([
      {
        byte_size: 72,
        created_at: 1_786_435_200,
        filename: '2026-08-11_09-00-00.md',
        id: '2026-08-11_09-00-00'
      }
    ])
    const onOpenOutput = vi.fn()

    render(
      <CronJobSidebarRuns
        jobId="report-job"
        noAgent
        onOpenOutput={onOpenOutput}
        onOpenSession={vi.fn()}
        profile="worker_alpha"
      />
    )

    fireEvent.click(await screen.findByRole('button'))

    expect(getCronJobOutputs).toHaveBeenCalledWith('report-job', 5, 'worker_alpha')
    expect(onOpenOutput).toHaveBeenCalledWith('report-job', '2026-08-11_09-00-00', 'worker_alpha')
    expect(getCronJobRuns).not.toHaveBeenCalled()
  })

  it('keeps agent-backed runs connected to their real conversation sessions', async () => {
    vi.mocked(getCronJobRuns).mockResolvedValue([
      {
        ended_at: null,
        id: 'cron_report-job_20260811_090000',
        input_tokens: 0,
        is_active: false,
        last_active: 1_786_435_200,
        message_count: 1,
        model: null,
        output_tokens: 0,
        preview: null,
        source: 'cron',
        started_at: 1_786_435_000,
        title: 'Report',
        tool_call_count: 0
      }
    ])
    const onOpenSession = vi.fn()

    render(
      <CronJobSidebarRuns
        jobId="report-job"
        noAgent={false}
        onOpenOutput={vi.fn()}
        onOpenSession={onOpenSession}
      />
    )

    fireEvent.click(await screen.findByRole('button'))

    expect(getCronJobRuns).toHaveBeenCalledWith('report-job', 5)
    expect(onOpenSession).toHaveBeenCalledWith('cron_report-job_20260811_090000')
    expect(getCronJobOutputs).not.toHaveBeenCalled()
  })

  it('falls back to durable output when an agent run has no stored session', async () => {
    vi.mocked(getCronJobRuns).mockResolvedValue([])
    vi.mocked(getCronJobOutputs).mockResolvedValue([
      {
        byte_size: 72,
        created_at: 1_786_435_200,
        filename: '2026-08-11_09-00-00.md',
        id: '2026-08-11_09-00-00'
      }
    ])
    const onOpenOutput = vi.fn()

    render(
      <CronJobSidebarRuns
        jobId="report-job"
        noAgent={false}
        onOpenOutput={onOpenOutput}
        onOpenSession={vi.fn()}
        profile="worker_alpha"
      />
    )

    fireEvent.click(await screen.findByRole('button'))

    expect(onOpenOutput).toHaveBeenCalledWith('report-job', '2026-08-11_09-00-00', 'worker_alpha')
  })

  it('does not describe a failed output request as an empty history', async () => {
    vi.mocked(getCronJobOutputs).mockRejectedValue(new Error('profile backend unavailable'))

    render(<CronJobSidebarRuns jobId="report-job" noAgent onOpenOutput={vi.fn()} onOpenSession={vi.fn()} />)

    expect(await screen.findByText(TRANSLATIONS.en.cron.failedLoad)).toBeTruthy()
    expect(screen.queryByText(TRANSLATIONS.en.cron.noRuns)).toBeNull()
  })
})
