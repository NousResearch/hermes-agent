import { act, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { SessionInfo } from '@/hermes'
import { I18nProvider } from '@/i18n'
import { $collapsedSessionDateGroups } from '@/store/session-date-group-collapse'

import type { SidebarProjectTree, SidebarSessionGroup } from './projects'
import { SidebarSessionsSection } from './sessions-section'

const session = (id: string, lastActive: number): SessionInfo =>
  ({
    active: false,
    id,
    last_active: lastActive,
    message_count: 1,
    source: 'cli',
    started_at: lastActive,
    title: id,
    tool_call_count: 0
  }) as unknown as SessionInfo

const renderSection = (sessions: SessionInfo[]) =>
  render(
    <I18nProvider configClient={null} initialLocale="en">
      <SidebarSessionsSection
        activeSessionId={null}
        dateGrouped
        dateGroupScope="test-scope"
        emptyState={null}
        label="Sessions"
        onArchiveSession={vi.fn()}
        onDeleteSession={vi.fn()}
        onResumeSession={vi.fn()}
        onToggle={vi.fn()}
        onTogglePin={vi.fn()}
        open
        pinned={false}
        sessions={sessions}
        workingSessionIdSet={new Set()}
      />
    </I18nProvider>
  )

const openDateGroupMenu = () => {
  const trigger = screen.getByRole('button', { name: 'Session date group actions' })

  fireEvent.pointerDown(trigger, { button: 0, pointerType: 'mouse' })
  fireEvent.pointerUp(trigger, { button: 0, pointerType: 'mouse' })
  fireEvent.click(trigger)
}

// Two repos, each with one main lane, so the entered-project view renders two
// independent `renderRowsDated` lists — one temporal group per lane.
const projectContentWithRepoSessions = (repoSessions: SessionInfo[][]): SidebarProjectTree => ({
  id: 'project-1',
  label: 'Project',
  path: '/project',
  repos: repoSessions.map((sessionsInRepo, index) => ({
    id: `repo-${index}`,
    label: `repo-${index}`,
    path: `/project/repo-${index}`,
    sessionCount: sessionsInRepo.length,
    groups: [
      {
        id: `repo-${index}::main`,
        isMain: true,
        label: 'main',
        path: `/project/repo-${index}`,
        sessions: sessionsInRepo
      }
    ]
  })),
  sessionCount: repoSessions.reduce((sum, sessionsInRepo) => sum + sessionsInRepo.length, 0)
})

const renderProjectContentSection = (projectContent: SidebarProjectTree) =>
  render(
    <I18nProvider configClient={null} initialLocale="en">
      <SidebarSessionsSection
        activeSessionId={null}
        dateGrouped
        dateGroupScope="test-scope"
        emptyState={null}
        label="Sessions"
        onArchiveSession={vi.fn()}
        onDeleteSession={vi.fn()}
        onResumeSession={vi.fn()}
        onToggle={vi.fn()}
        onTogglePin={vi.fn()}
        open
        pinned={false}
        projectContent={projectContent}
        sessions={[]}
        workingSessionIdSet={new Set()}
      />
    </I18nProvider>
  )

// A genuine profile/source group, as `showAllProfiles` passes via `groups`
// (see `displayAgentGroups`/`profileGroups` in sidebar/index.tsx) — a real
// backend profile lane, not a project/worktree lane.
const profileGroup = (id: string, sessions: SessionInfo[]): SidebarSessionGroup => ({
  id,
  label: id,
  mode: 'profile',
  path: null,
  sessions
})

const renderGroupsSection = (groups: SidebarSessionGroup[]) =>
  render(
    <I18nProvider configClient={null} initialLocale="en">
      <SidebarSessionsSection
        activeSessionId={null}
        dateGrouped
        dateGroupScope="test-scope"
        emptyState={null}
        groups={groups}
        label="Sessions"
        onArchiveSession={vi.fn()}
        onDeleteSession={vi.fn()}
        onResumeSession={vi.fn()}
        onToggle={vi.fn()}
        onTogglePin={vi.fn()}
        open
        pinned={false}
        sessions={[]}
        workingSessionIdSet={new Set()}
      />
    </I18nProvider>
  )

describe('SidebarSessionsSection date groups', () => {
  beforeEach(() => {
    $collapsedSessionDateGroups.set({})
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it('moves an unchanged Today group to Yesterday when local midnight passes', () => {
    vi.useFakeTimers()
    vi.setSystemTime(new Date(2026, 6, 31, 23, 59, 59))

    renderSection([session('Midnight session', new Date(2026, 6, 31, 12).getTime() / 1000)])

    expect(screen.getByRole('button', { name: 'Collapse Today' })).toBeTruthy()

    act(() => {
      vi.advanceTimersByTime(1_000)
    })

    expect(screen.queryByRole('button', { name: 'Collapse Today' })).toBeNull()
    expect(screen.getByRole('button', { name: 'Collapse Yesterday' })).toBeTruthy()
    expect(screen.getByText('Midnight session')).toBeTruthy()
  })

  it('collapses and restores a group from its accessible date disclosure', () => {
    renderSection([session('Today session', Math.floor(Date.now() / 1000))])

    expect(screen.getByText('Today session')).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: 'Collapse Today' }))

    expect(screen.queryByText('Today session')).toBeNull()
    expect($collapsedSessionDateGroups.get()['test-scope']).toHaveLength(1)

    fireEvent.click(screen.getByRole('button', { name: 'Expand Today' }))

    expect(screen.getByText('Today session')).toBeTruthy()
    expect($collapsedSessionDateGroups.get()['test-scope']).toBeUndefined()
  })

  it('offers collapse-all and expand-all from the Sessions kebab menu', async () => {
    renderSection([
      session('Today session', Math.floor(Date.now() / 1000)),
      session('Old session', Math.floor(new Date(2025, 0, 10).getTime() / 1000))
    ])

    openDateGroupMenu()
    fireEvent.click(await screen.findByText('Collapse all date groups'))

    expect(screen.queryByText('Today session')).toBeNull()
    expect(screen.queryByText('Old session')).toBeNull()
    expect($collapsedSessionDateGroups.get()['test-scope']).toHaveLength(2)

    openDateGroupMenu()
    fireEvent.click(await screen.findByText('Expand all date groups'))

    expect(screen.getByText('Today session')).toBeTruthy()
    expect(screen.getByText('Old session')).toBeTruthy()
  })

  it('offers collapse-all and expand-all across every nested project lane', async () => {
    const projectContent = projectContentWithRepoSessions([
      [session('Repo A today session', Math.floor(Date.now() / 1000))],
      [session('Repo B old session', Math.floor(new Date(2025, 0, 10).getTime() / 1000))]
    ])

    renderProjectContentSection(projectContent)

    expect(screen.getByText('Repo A today session')).toBeTruthy()
    expect(screen.getByText('Repo B old session')).toBeTruthy()

    openDateGroupMenu()
    fireEvent.click(await screen.findByText('Collapse all date groups'))

    expect(screen.queryByText('Repo A today session')).toBeNull()
    expect(screen.queryByText('Repo B old session')).toBeNull()
    expect($collapsedSessionDateGroups.get()['test-scope']).toHaveLength(2)

    openDateGroupMenu()
    fireEvent.click(await screen.findByText('Expand all date groups'))

    expect(screen.getByText('Repo A today session')).toBeTruthy()
    expect(screen.getByText('Repo B old session')).toBeTruthy()
    expect($collapsedSessionDateGroups.get()['test-scope']).toBeUndefined()
  })

  it('offers collapse-all and expand-all across every genuine profile/source group', async () => {
    const groups = [
      profileGroup('profile-a', [session('Profile A today session', Math.floor(Date.now() / 1000))]),
      profileGroup('profile-b', [session('Profile B old session', Math.floor(new Date(2025, 0, 10).getTime() / 1000))])
    ]

    const { container } = renderGroupsSection(groups)

    const groupHeadingOrder = () =>
      [...container.querySelectorAll('[title="profile-a"], [title="profile-b"]')].map(el => el.getAttribute('title'))

    // Each group renders its own temporal date group (not a flat static list).
    expect(screen.getByRole('button', { name: 'Collapse Today' })).toBeTruthy()
    expect(screen.getByText('Profile A today session')).toBeTruthy()
    expect(screen.getByText('Profile B old session')).toBeTruthy()
    expect(groupHeadingOrder()).toEqual(['profile-a', 'profile-b'])

    openDateGroupMenu()
    fireEvent.click(await screen.findByText('Collapse all date groups'))

    expect(screen.queryByText('Profile A today session')).toBeNull()
    expect(screen.queryByText('Profile B old session')).toBeNull()
    expect($collapsedSessionDateGroups.get()['test-scope']).toHaveLength(2)

    // Parent profile headings stay mounted, in the same order, untouched by
    // collapsing their conversations' date groups.
    expect(groupHeadingOrder()).toEqual(['profile-a', 'profile-b'])

    openDateGroupMenu()
    fireEvent.click(await screen.findByText('Expand all date groups'))

    expect(screen.getByText('Profile A today session')).toBeTruthy()
    expect(screen.getByText('Profile B old session')).toBeTruthy()
    expect($collapsedSessionDateGroups.get()['test-scope']).toBeUndefined()
  })

  it('still offers collapse-all and expand-all for the flat (non-project) list', async () => {
    renderSection([session('Flat today session', Math.floor(Date.now() / 1000))])

    openDateGroupMenu()
    fireEvent.click(await screen.findByText('Collapse all date groups'))

    expect(screen.queryByText('Flat today session')).toBeNull()

    openDateGroupMenu()
    fireEvent.click(await screen.findByText('Expand all date groups'))

    expect(screen.getByText('Flat today session')).toBeTruthy()
  })

  it('does not offer the date group menu for a projectOverview with no temporal group', () => {
    render(
      <I18nProvider configClient={null} initialLocale="en">
        <SidebarSessionsSection
          activeSessionId={null}
          dateGrouped
          dateGroupScope="test-scope"
          emptyState={null}
          label="Sessions"
          onArchiveSession={vi.fn()}
          onDeleteSession={vi.fn()}
          onResumeSession={vi.fn()}
          onToggle={vi.fn()}
          onTogglePin={vi.fn()}
          open
          pinned={false}
          projectOverview={[
            {
              id: 'project-1',
              label: 'Project',
              path: '/project',
              repos: [],
              sessionCount: 0
            }
          ]}
          sessions={[]}
          workingSessionIdSet={new Set()}
        />
      </I18nProvider>
    )

    expect(screen.queryByRole('button', { name: 'Session date group actions' })).toBeNull()
  })
})
