// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { group } from '@/components/pane-shell/tree/model'
import { $layoutTree, noteActiveTreeGroup } from '@/components/pane-shell/tree/store'
import { SidebarProvider } from '@/components/ui/sidebar'
import { registry } from '@/contrib/registry'
import { $cronJobs } from '@/store/cron'
import { $selectedStoredSessionId, $sessions } from '@/store/session'
import { makeSessionInfo } from '@/test/session-info'

import { type AppView, ROUTES_AREA, SIDEBAR_NAV_AREA } from '../../routes'

import { ChatSidebar } from './index'

// The below-Sessions stack (messaging platforms + cron jobs) renders under one
// master collapse so the flex-1 Sessions list can reclaim the space in one
// click instead of collapsing each section individually (#105234).
const noop = () => {}

const noopAsync = async () => {}

const sessionRows = [
  makeSessionInfo({ id: 'sess-one', last_active: 2, profile: 'default', started_at: 1, title: 'Session one' })
]

const cronJobs = [
  {
    created_at: '2026-09-18T00:00:00Z',
    cron: '0 9 * * *',
    enabled: true,
    id: 'job-1',
    last_run_at: null,
    message: 'Daily report',
    name: 'Daily report',
    next_run_at: '2026-09-19T09:00:00Z',
    paused: false,
    profile: 'default',
    session_id: null,
    source: 'user',
    state: 'active',
    timezone: 'UTC',
    user_id: null
  }
]

const renderSidebar = () =>
  render(
    <MemoryRouter initialEntries={['/workspace']}>
      <SidebarProvider>
        <ChatSidebar
          currentView={'chat' as AppView}
          onArchiveSession={noop}
          onBranchSession={noop}
          onDeleteSession={noop}
          onLoadMoreSessions={noop}
          onManageCronJob={noop}
          onNavigate={noop}
          onNewSessionInWorkspace={noop}
          onNewSessionSplit={noop}
          onResumeSession={noop}
          onTriggerCronJob={noopAsync}
        />
      </SidebarProvider>
    </MemoryRouter>
  )

describe('ChatSidebar below-Sessions master collapse', () => {
  let disposeContributions: () => void

  beforeEach(() => {
    disposeContributions = registry.registerMany([
      { area: ROUTES_AREA, id: 'kanban-page', data: { path: '/kanban' }, render: () => null },
      { area: SIDEBAR_NAV_AREA, id: 'kanban-nav', data: { codicon: 'project', label: 'Kanban', path: '/kanban' } }
    ])
    $selectedStoredSessionId.set('sess-one')
    $sessions.set(sessionRows)
    $cronJobs.set(cronJobs as never)
    $layoutTree.set(
      group(['workspace'], { active: 'workspace', id: 'workspace-group' })
    )
    noteActiveTreeGroup('workspace-group')
  })

  afterEach(() => {
    cleanup()
    disposeContributions()
    $selectedStoredSessionId.set(null)
    $sessions.set([])
    $cronJobs.set([])
    $layoutTree.set(null)
    noteActiveTreeGroup(null)
    localStorage.clear()
  })

  it('renders the messaging & jobs stack open by default and collapses it in one click', () => {
    renderSidebar()

    // Open by default: the cron section (a child of the stack) is present.
    expect(screen.getByText('Cron jobs')).toBeTruthy()

    // One click on the aggregate header collapses the whole stack.
    fireEvent.click(screen.getByRole('button', { name: 'Toggle messaging and cron sections' }))

    expect(screen.queryByText('Cron jobs')).toBeNull()

    // And back open.
    fireEvent.click(screen.getByRole('button', { name: 'Toggle messaging and cron sections' }))
    expect(screen.getByText('Cron jobs')).toBeTruthy()
  })

  it('keeps the Sessions list visible while the below-Sessions stack is collapsed', () => {
    renderSidebar()

    fireEvent.click(screen.getByRole('button', { name: 'Toggle messaging and cron sections' }))

    expect(screen.queryByText('Cron jobs')).toBeNull()
    expect(screen.getByText('Session one')).toBeTruthy()
  })
})
