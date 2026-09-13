import { cleanup, render, screen } from '@testing-library/react'
import type * as React from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { CronJob } from '@/types/hermes'

import { SidebarCronJobsSection } from './cron-jobs-section'

afterEach(cleanup)

vi.mock('@hermes/shared', () => ({
  createCronTriggerController: () => ({ run: vi.fn() })
}))
vi.mock('@nanostores/react', () => ({ useStore: () => false }))
vi.mock('@/components/pane-shell/pane-visibility', () => ({ usePaneVisible: () => false }))
vi.mock('@/components/ui/actions-menu', () => ({
  ActionsContextMenu: ({ children }: { children: React.ReactNode }) => <>{children}</>,
  renderActionItem: vi.fn()
}))
vi.mock('@/components/ui/codicon', () => ({ Codicon: () => null }))
vi.mock('@/components/ui/disclosure-caret', () => ({ DisclosureCaret: () => null }))
vi.mock('@/components/ui/glyph-spinner', () => ({ GlyphSpinner: () => null }))
vi.mock('@/components/ui/sidebar', () => ({
  SidebarGroup: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  SidebarGroupContent: ({ children }: { children: React.ReactNode }) => <div>{children}</div>
}))
vi.mock('@/components/ui/tooltip', () => ({ Tip: ({ children }: { children: React.ReactNode }) => <>{children}</> }))
vi.mock('@/hermes', () => ({
  deleteCronJob: vi.fn(),
  getCronJobRuns: vi.fn(),
  pauseCronJob: vi.fn(),
  resumeCronJob: vi.fn()
}))
vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      common: { delete: 'Delete' },
      cron: {
        actionsTitle: 'Cron actions',
        deleteDescPrefix: '',
        deleteDescSuffix: '',
        deleteTitle: 'Delete cron job',
        deleted: 'Deleted',
        failedDelete: 'Delete failed',
        failedUpdate: 'Update failed',
        hideRuns: 'Hide runs',
        manage: 'Manage',
        pause: 'Pause',
        paused: 'Paused',
        resume: 'Resume',
        resumed: 'Resumed',
        showRuns: 'Show runs',
        states: {},
        triggerNow: 'Trigger now'
      }
    }
  })
}))
vi.mock('@/lib/time', () => ({ fmtDayTime: { format: () => '' }, relativeTime: () => 'soon' }))
vi.mock('@/store/confirm', () => ({ confirm: vi.fn() }))
vi.mock('@/store/cron', () => ({ updateCronJobs: vi.fn() }))
vi.mock('@/store/live-sync', () => ({ $changeEventsAvailable: {}, $cronChangeTick: {} }))
vi.mock('@/store/notifications', () => ({ notify: vi.fn(), notifyError: vi.fn() }))
vi.mock('@/store/session', () => ({ $selectedStoredSessionId: {} }))
vi.mock('../../cron/job-state', () => ({
  jobState: () => 'running',
  jobTitle: (job: { name: string }) => job.name,
  STATE_DOT: { running: 'bg-green-500' }
}))
vi.mock('../../shell/sidebar-label', () => ({
  SidebarPanelLabel: ({ children }: { children: React.ReactNode }) => <span>{children}</span>
}))
vi.mock('./chrome', () => ({
  SidebarRowBody: ({ children, ...props }: React.ComponentProps<'button'>) => <button {...props}>{children}</button>,
  SidebarRowLabel: ({ children, className }: React.ComponentProps<'span'>) => (
    <span className={className}>{children}</span>
  ),
  SidebarRowLead: ({ children }: { children: React.ReactNode }) => <span>{children}</span>,
  SidebarRowShell: ({ children }: { children: React.ReactNode }) => <div>{children}</div>
}))
vi.mock('./load-more-row', () => ({ SidebarLoadMoreRow: () => null }))

describe('SidebarCronJobsSection', () => {
  it('keeps cron job labels explicitly single-line truncated', () => {
    const label = 'A cron job title that must not make sidebar rows wrap'
    const job = { id: 'cron-1', name: label } as CronJob

    render(
      <SidebarCronJobsSection
        jobs={[job]}
        label="Cron jobs"
        onManageJob={vi.fn()}
        onOpenRun={vi.fn()}
        onToggle={vi.fn()}
        onTriggerJob={async () => {}}
        open
      />
    )

    expect(screen.getByText(label).className).toContain('truncate')
  })
})
