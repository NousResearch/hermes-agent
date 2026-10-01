import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider, TRANSLATIONS } from '@/i18n'
import { fmtDayTime } from '@/lib/time'
import type { CronJob, SessionInfo } from '@/types/hermes'

import { SidebarCronJobsSection } from './cron-jobs-section'

// Hoisted so the vi.mock factory (also hoisted) can close over it.
const { getCronJobRuns } = vi.hoisted(() => ({ getCronJobRuns: vi.fn() }))

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<object>()),
  getCronJobRuns
}))

// A cron run row as the backend serves it: the job's owning profile plus the
// connection tag the desktop stamps on backend-returned session pages.
const RUN = {
  connection_id: 'gw-tailscale',
  ended_at: null,
  id: 'cron-nightly-1',
  input_tokens: 0,
  is_active: false,
  last_active: 1_700_000_000,
  message_count: 2,
  model: null,
  output_tokens: 0,
  profile: 'research',
  source: 'cron'
} as SessionInfo

const JOB = { enabled: true, id: 'job-1', name: 'nightly', schedule: '* * * * *', state: 'scheduled' } as CronJob

beforeEach(() => {
  getCronJobRuns.mockResolvedValue([RUN])
})

afterEach(() => {
  cleanup()
  getCronJobRuns.mockReset()
})

function renderSection(onOpenRun: (sessionId: string, session?: SessionInfo) => void) {
  return render(
    <I18nProvider configClient={null} initialLocale="en">
      <SidebarCronJobsSection
        jobs={[JOB]}
        label="Cron jobs"
        onManageJob={() => {}}
        onOpenRun={onOpenRun}
        onToggle={() => {}}
        onTriggerJob={async () => {}}
        open
      />
    </I18nProvider>
  )
}

describe('SidebarCronJobsSection run rows', () => {
  // Contract (#82527): clicking a run must hand the open path the run ROW, not
  // its bare id. The row carries the owning (connection, profile); without it
  // the resume cannot name the backend that holds the run and falls to the
  // ambient one — so an SSH/remote run's transcript never loads.
  it('opens a run with the run ROW so the resume can pin its owning backend', async () => {
    const onOpenRun = vi.fn()
    renderSection(onOpenRun)

    fireEvent.click(screen.getByRole('button', { name: TRANSLATIONS.en.cron.showRuns }))

    const runButton = await screen.findByRole('button', {
      name: fmtDayTime.format(new Date(RUN.last_active * 1000))
    })

    fireEvent.click(runButton)

    expect(onOpenRun).toHaveBeenCalledWith(RUN.id, RUN)
  })
})

// Session-title wrapping must not change cron row height.
describe('cron label overflow', () => {
  it('keeps cron labels explicitly single-line truncated', () => {
    renderSection(vi.fn())
    expect(screen.getByText('nightly').className).toContain('truncate')
  })
})
