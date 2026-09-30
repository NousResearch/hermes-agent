import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { en } from '@/i18n/en'
import { $cronJobs } from '@/store/cron'
import type { CronJob } from '@/types/hermes'

import { CronView } from './index'

// The detail pane fetches runs and delivery targets on mount; both are irrelevant
// to the copy affordance, so stub the transport the cron API calls ride on.
vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  getCronJobRuns: vi.fn(async () => []),
  getCronDeliveryTargets: vi.fn(async () => ({ targets: [] })),
  getAutomationBlueprints: vi.fn(async () => ({ blueprints: [] }))
}))

afterEach(cleanup)

const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })

function renderCronView() {
  render(
    <QueryClientProvider client={queryClient}>
      <CronView onClose={() => {}} />
    </QueryClientProvider>
  )
}

function makeJob(overrides: Partial<CronJob> = {}): CronJob {
  return {
    deliver: 'local',
    enabled: true,
    id: 'job-1',
    name: 'Morning briefing',
    prompt: 'Summarize my unread Slack threads',
    schedule: { display: 'Daily at 9:00 AM', expr: '0 9 * * *' },
    state: 'scheduled',
    ...overrides
  }
}

describe('CronJobDetail copy prompt', () => {
  beforeEach(() => {
    $cronJobs.set([makeJob()])
  })

  it('copies the job prompt through the desktop clipboard bridge', async () => {
    const writeClipboard = vi.fn().mockResolvedValue(undefined)

    ;(window as unknown as { hermesDesktop?: unknown }).hermesDesktop = { writeClipboard }

    renderCronView()

    const button = screen.getByRole('button', { name: en.cron.copyPrompt })
    await fireEvent.click(button)

    expect(writeClipboard).toHaveBeenCalledWith('Summarize my unread Slack threads')

    delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
  })

  it('renders no copy affordance for a script-only job', () => {
    $cronJobs.set([makeJob({ no_agent: true, prompt: '', script: 'uptime' })])
    // No prompt AND no script copy: a script-only job's block is its script, not a prompt.

    renderCronView()

    expect(screen.queryByRole('button', { name: en.cron.copyPrompt })).toBeNull()
  })
})
