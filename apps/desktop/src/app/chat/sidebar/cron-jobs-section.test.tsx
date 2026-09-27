import { cleanup, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { fmtDayTime } from '@/lib/time'
import { $readOnlyCronRuns, isStoredTranscriptReadOnly } from '@/store/read-only-transcript'
import type { CronJob, SessionInfo } from '@/types/hermes'

import { SidebarCronJobsSection } from './cron-jobs-section'

// The peek's run list comes off the backend; the liveness flags below are the
// endpoint's own (`hermes_cli/web_routers/cron.py` computes `is_active` as
// `ended_at IS NULL` + a recent-activity window).
const getCronJobRuns = vi.fn<() => Promise<SessionInfo[]>>()

vi.mock('@/hermes', async importOriginal => {
  const actual = (await importOriginal()) as Record<string, unknown>

  return {
    ...actual,
    deleteCronJob: vi.fn(),
    getCronJobRuns: (...args: unknown[]) => getCronJobRuns(...(args as [])),
    pauseCronJob: vi.fn(),
    resumeCronJob: vi.fn()
  }
})

const job: CronJob = { enabled: true, id: 'job-1', name: 'Daily digest', schedule_display: '30 8 * * *' }

const run = (over: Partial<SessionInfo>): SessionInfo =>
  ({
    ended_at: 1_700_000_600,
    id: 'cron_job-1_1700000000',
    is_active: false,
    last_active: 1_700_000_600,
    source: 'cron',
    started_at: 1_700_000_000,
    ...over
  }) as SessionInfo

const runLabel = (row: SessionInfo) => fmtDayTime.format(new Date((row.last_active || row.started_at) * 1000))

beforeEach(() => {
  $readOnlyCronRuns.set(new Set())
  getCronJobRuns.mockReset()
})

afterEach(cleanup)

async function renderRunsPeek(rows: SessionInfo[]) {
  getCronJobRuns.mockResolvedValue(rows)

  const onOpenRun = vi.fn()

  render(
    <SidebarCronJobsSection
      jobs={[job]}
      label="Cron Jobs"
      onManageJob={vi.fn()}
      onOpenRun={onOpenRun}
      onToggle={vi.fn()}
      onTriggerJob={vi.fn()}
      open
    />
  )

  // Expand the job row's run peek.
  screen.getByRole('button', { name: 'Show runs' }).click()

  return { onOpenRun }
}

describe('SidebarCronJobsSection run rows', () => {
  // Contract (#82527): clicking a run must hand the open path the run ROW, not
  // its bare id. The row carries the owning (connection, profile); without it
  // the resume cannot name the backend that holds the run and falls to the
  // ambient one — so an SSH/remote run's transcript never loads.
  it('opens a run with the run ROW so the resume can pin its owning backend', async () => {
    const remote = run({ connection_id: 'gw-tailscale', id: 'cron_remote', profile: 'research' })
    const { onOpenRun } = await renderRunsPeek([remote])

    const row = await screen.findByRole('button', { name: runLabel(remote) })
    row.click()

    expect(onOpenRun).toHaveBeenCalledWith('cron_remote', remote)
  })
})

describe('SidebarCronJobsSection run peek — zombie cron runs (#88443)', () => {
  it('opens a never-closed run (ended_at NULL, not live) READ-ONLY', async () => {
    const zombie = run({ ended_at: null, id: 'cron_zombie', is_active: false })
    const { onOpenRun } = await renderRunsPeek([zombie])

    const closeable = await screen.findByRole('button', { name: runLabel(zombie) })
    closeable.click()

    // The click still opens the run — its output stays one click away …
    expect(onOpenRun).toHaveBeenCalledWith('cron_zombie', zombie)
    // … but the composer can no longer route a send into the dead session.
    expect(isStoredTranscriptReadOnly('cron_zombie')).toBe(true)
  })

  it('leaves a properly closed run writable', async () => {
    const closed = run({ ended_at: 1_700_000_600, id: 'cron_closed', is_active: false })
    const { onOpenRun } = await renderRunsPeek([closed])

    const row = await screen.findByRole('button', { name: runLabel(closed) })
    row.click()

    expect(onOpenRun).toHaveBeenCalledWith('cron_closed', closed)
    expect(isStoredTranscriptReadOnly('cron_closed')).toBe(false)
  })

  it('leaves a live run writable', async () => {
    const live = run({ ended_at: null, id: 'cron_live', is_active: true })
    const { onOpenRun } = await renderRunsPeek([live])

    const row = await screen.findByRole('button', { name: runLabel(live) })
    row.click()

    expect(onOpenRun).toHaveBeenCalledWith('cron_live', live)
    expect(isStoredTranscriptReadOnly('cron_live')).toBe(false)
  })

  it('does not latch anything merely by rendering the peek', async () => {
    await renderRunsPeek([run({ ended_at: null, id: 'cron_zombie', is_active: false })])

    await screen.findByRole('button', { name: runLabel(run({ ended_at: null, id: 'cron_zombie' })) })

    expect(isStoredTranscriptReadOnly('cron_zombie')).toBe(false)
  })

  it('still shows the run time for script-only output rows', async () => {
    const output = run({ ended_at: null, id: 'cron_output:job-1:latest', is_active: false, source: 'cron_output' })

    await renderRunsPeek([output])

    await waitFor(() => expect(screen.getByText(runLabel(output))).toBeTruthy())
    // No chat affordance to click, and nothing latched.
    expect(screen.queryByRole('button', { name: runLabel(output) })).toBeNull()
    expect(isStoredTranscriptReadOnly(output.id)).toBe(false)
  })
})
