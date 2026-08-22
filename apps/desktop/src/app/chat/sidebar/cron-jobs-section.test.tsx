import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import type * as React from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { CronJob } from '@/types/hermes'

import { SidebarCronJobsSection } from './cron-jobs-section'

afterEach(cleanup)

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      common: { delete: 'Delete' },
      sidebar: { loadCount: (n: number) => `Show ${n} more`, loadMore: 'Show more', loading: 'Loading…' },
      cron: {
        actionsTitle: 'Actions',
        deleteDescPrefix: 'Delete ',
        deleteDescSuffix: '?',
        deleteTitle: 'Delete job',
        deleted: 'Deleted',
        failedDelete: 'Delete failed',
        failedUpdate: 'Update failed',
        hideRuns: 'Hide runs',
        loading: 'Loading',
        manage: 'Manage',
        noRuns: 'No runs',
        pause: 'Pause',
        paused: 'Paused',
        resume: 'Resume',
        resumed: 'Resumed',
        showRuns: 'Show runs',
        shownOf: (shown: number, total: number) => `${shown} of ${total}`,
        states: { scheduled: 'scheduled' },
        triggerNow: 'Run now'
      }
    }
  })
}))

vi.mock('@/components/pane-shell/pane-visibility', () => ({ usePaneVisible: () => true }))
vi.mock('@nanostores/react', () => ({ useStore: () => null }))
vi.mock('@/store/cron', () => ({ updateCronJobs: vi.fn() }))
vi.mock('@/store/live-sync', () => ({ $changeEventsAvailable: {}, $cronChangeTick: {} }))
vi.mock('@/store/session', () => ({ $selectedStoredSessionId: {} }))
vi.mock('@/hermes', () => ({
  deleteCronJob: vi.fn(),
  getCronJobRuns: vi.fn(),
  pauseCronJob: vi.fn(),
  resumeCronJob: vi.fn()
}))
vi.mock('@/store/confirm', () => ({ confirm: vi.fn() }))
vi.mock('@hermes/shared', () => ({
  createCronTriggerController: () => ({ run: vi.fn() })
}))
vi.mock('@/components/ui/actions-menu', () => ({
  ActionsContextMenu: ({ children }: { children: React.ReactNode }) => <>{children}</>,
  renderActionItem: () => null
}))
vi.mock('@/components/ui/codicon', () => ({ Codicon: () => null }))
vi.mock('@/components/ui/disclosure-caret', () => ({ DisclosureCaret: () => null }))
vi.mock('@/components/ui/glyph-spinner', () => ({ GlyphSpinner: () => null }))
vi.mock('@/components/ui/sidebar', () => ({
  SidebarGroup: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  SidebarGroupContent: ({ children }: { children: React.ReactNode }) => <div>{children}</div>
}))
vi.mock('@/components/ui/tooltip', () => ({ Tip: ({ children }: { children: React.ReactNode }) => <>{children}</> }))
vi.mock('../../shell/sidebar-label', () => ({
  SidebarPanelLabel: ({ children }: { children: React.ReactNode }) => <span>{children}</span>
}))
vi.mock('./chrome', () => ({
  SidebarRowBody: ({ children, ...props }: { children: React.ReactNode }) => <div {...props}>{children}</div>,
  SidebarRowLabel: ({ children }: { children: React.ReactNode }) => <span>{children}</span>,
  SidebarRowLead: ({ children }: { children: React.ReactNode }) => <span>{children}</span>,
  SidebarRowShell: ({ children, actions }: { children: React.ReactNode; actions: React.ReactNode }) => (
    <div>{children}{actions}</div>
  )
}))

function makeJob(id: string) {
  return { enabled: true, id, name: id, next_run_at: '2099-01-01T00:00:00Z', state: 'scheduled' } as unknown as CronJob
}

describe('SidebarCronJobsSection load-more affordance', () => {
  it('shows the hidden count, renders a text row, and reveals the hidden job', () => {
    render(
      <SidebarCronJobsSection
        jobs={[
          makeJob('job-1'),
          makeJob('job-2'),
          makeJob('job-3'),
          makeJob('job-4'),
          makeJob('job-5'),
          makeJob('job-6'),
          makeJob('job-7')
        ]}
        label="Scheduled jobs"
        onManageJob={vi.fn()}
        onOpenRun={vi.fn()}
        onToggle={vi.fn()}
        onTriggerJob={vi.fn()}
        open
      />
    )

    expect(screen.getByText('Scheduled jobs')).toBeInTheDocument()
    expect(screen.getByText('6 of 7', { exact: false })).toBeInTheDocument()
    const loadMore = screen.getByRole('button', { name: 'Show 1 more…' })
    expect(loadMore).toHaveTextContent('Show 1 more…')
    expect(screen.queryByText('job-7')).not.toBeInTheDocument()

    fireEvent.click(loadMore)

    expect(screen.getByText('job-7')).toBeInTheDocument()
    // Once every job is shown, the "N of M" badge and the load-more control
    // must both disappear — the truncation affordance is gone.
    expect(screen.queryByText('6 of 7', { exact: false })).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Show 1 more…' })).not.toBeInTheDocument()
  })

  it('renders no count badge and no load-more row when all jobs already fit', () => {
    // INITIAL_VISIBLE_JOBS is 6, so 6 jobs or fewer should show fully with
    // neither the "N of M" badge nor the load-more control.
    render(
      <SidebarCronJobsSection
        jobs={[
          makeJob('job-1'),
          makeJob('job-2'),
          makeJob('job-3'),
          makeJob('job-4'),
          makeJob('job-5'),
          makeJob('job-6')
        ]}
        label="Scheduled jobs"
        onManageJob={vi.fn()}
        onOpenRun={vi.fn()}
        onToggle={vi.fn()}
        onTriggerJob={vi.fn()}
        open
      />
    )

    expect(screen.getByText('job-6')).toBeInTheDocument()
    expect(screen.queryByText(' of ', { exact: false })).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /more/i })).not.toBeInTheDocument()
  })
})
