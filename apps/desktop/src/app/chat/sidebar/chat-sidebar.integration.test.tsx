// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { group, split } from '@/components/pane-shell/tree/model'
import { $layoutTree, noteActiveTreeGroup } from '@/components/pane-shell/tree/store'
import { SidebarProvider } from '@/components/ui/sidebar'
import { registry } from '@/contrib/registry'
import { $sidebarProjectFilter, setSidebarGrouping, setSidebarRecentsOpen } from '@/store/layout'
import { $projectScope, $projectTree, ALL_PROJECTS } from '@/store/projects'
import { $selectedStoredSessionId, $sessions } from '@/store/session'
import { $removedSessionIds } from '@/store/session-removal'
import { makeSessionInfo } from '@/test/session-info'

import { type AppView, ROUTES_AREA, SIDEBAR_NAV_AREA } from '../../routes'

import type { SidebarProjectTree } from './projects'

import { ChatSidebar } from './index'

const noop = () => {}

const noopAsync = async () => {}

const sessionRows = [
  makeSessionInfo({ id: 'tile-one', last_active: 2, profile: 'default', started_at: 1, title: 'Tile one' }),
  makeSessionInfo({ id: 'tile-two', last_active: 2, profile: 'default', started_at: 1, title: 'Tile two' })
]

const renderSidebar = (pathname: string, currentView: AppView, onNewSessionInWorkspace = noop) =>
  render(
    <MemoryRouter initialEntries={[pathname]}>
      <SidebarProvider>
        <ChatSidebar
          currentView={currentView}
          onArchiveSession={noop}
          onBranchSession={noop}
          onDeleteSession={noop}
          onLoadMoreSessions={noop}
          onManageCronJob={noop}
          onNavigate={noop}
          onNewSessionInWorkspace={onNewSessionInWorkspace}
          onNewSessionSplit={noop}
          onResumeSession={noop}
          onTriggerCronJob={noopAsync}
        />
      </SidebarProvider>
    </MemoryRouter>
  )

const currentButtons = () =>
  screen.queryAllByRole('button').filter(button => button.classList.contains('bg-(--ui-control-active-background)'))

const expectOnlyCurrent = (label: string | null) => {
  const button = label ? screen.getByRole('button', { name: label }) : null

  expect(currentButtons()).toEqual(button ? [button] : [])
}

const expectOnlySelectedSession = (title: string | null) => {
  const rows = ['Tile one', 'Tile two']
    .map(label => screen.queryByText(label)?.closest('.group.row-hover'))
    .filter(row => row !== undefined)

  const selectedRows = rows.filter(row => row?.className.includes('bg-(--ui-row-active-background)'))
  const expected = title ? [screen.getByText(title).closest('.group.row-hover')] : []

  expect(selectedRows).toEqual(expected)
}

const focus = (groupId: null | string) => act(() => noteActiveTreeGroup(groupId))

describe('ChatSidebar navigation activity', () => {
  let disposeContributions: () => void

  beforeEach(() => {
    disposeContributions = registry.registerMany([
      { area: ROUTES_AREA, id: 'kanban-page', data: { path: '/kanban' }, render: () => null },
      { area: ROUTES_AREA, id: 'reports-page', data: { path: '/reports' }, render: () => null },
      { area: SIDEBAR_NAV_AREA, id: 'kanban-nav', data: { codicon: 'project', label: 'Kanban', path: '/kanban' } },
      { area: SIDEBAR_NAV_AREA, id: 'reports-nav', data: { codicon: 'graph', label: 'Reports', path: '/reports' } }
    ])
    $selectedStoredSessionId.set('tile-one')
    $sessions.set(sessionRows)
    $projectScope.set(ALL_PROJECTS)
    $projectTree.set([
      {
        id: '__no_project__',
        isNoProject: true,
        label: 'Home',
        path: null,
        previewSessions: [],
        repos: [],
        sessionCount: 0
      } as SidebarProjectTree,
      {
        id: 'p_project',
        label: 'Project One',
        path: '/repos/project-one',
        previewSessions: [],
        repos: [],
        sessionCount: 0
      } as SidebarProjectTree
    ])
    setSidebarGrouping('date')
    setSidebarRecentsOpen(true)
    $sidebarProjectFilter.set([])
    $removedSessionIds.set(new Set())
    $layoutTree.set(
      split('row', [
        group(['workspace'], { active: 'workspace', id: 'workspace-group' }),
        group(['session-tile:tile-one'], { active: 'session-tile:tile-one', id: 'tile-one-group' }),
        group(['session-tile:tile-two'], { active: 'session-tile:tile-two', id: 'tile-two-group' })
      ])
    )
    noteActiveTreeGroup('workspace-group')
  })

  afterEach(() => {
    cleanup()
    disposeContributions()
    $selectedStoredSessionId.set(null)
    $sessions.set([])
    $projectScope.set(ALL_PROJECTS)
    $projectTree.set([])
    setSidebarGrouping('date')
    setSidebarRecentsOpen(true)
    $sidebarProjectFilter.set([])
    $removedSessionIds.set(new Set())
    $layoutTree.set(null)
    noteActiveTreeGroup(null)
  })

  it('keeps navigation and session activity coherent with the focused pane', () => {
    renderSidebar('/kanban', 'extension')
    expectOnlyCurrent('Kanban')
    expectOnlySelectedSession(null)

    focus('tile-one-group')
    expectOnlyCurrent(null)
    expectOnlySelectedSession('Tile one')

    focus('tile-two-group')
    expectOnlyCurrent(null)
    expectOnlySelectedSession('Tile two')

    focus(null)
    expectOnlyCurrent('Kanban')
    expectOnlySelectedSession(null)

    focus('tile-two-group')
    act(() => {
      $removedSessionIds.set(new Set(['tile-two']))
      $sessions.set([sessionRows[0]])
    })
    expectOnlyCurrent(null)
    expectOnlySelectedSession(null)

    act(() => {
      $removedSessionIds.set(new Set())
      $sessions.set(sessionRows)
    })

    for (const [pathname, currentView, label] of [
      ['/skills', 'skills', 'Capabilities'],
      ['/messaging', 'messaging', 'Messaging'],
      ['/artifacts', 'artifacts', 'Artifacts'],
      ['/cron', 'cron', 'Scheduled jobs']
    ] as const) {
      cleanup()
      focus('workspace-group')
      renderSidebar(pathname, currentView)
      expectOnlyCurrent(label)
      expectOnlySelectedSession(null)

      focus('tile-one-group')
      expectOnlyCurrent(null)
      expectOnlySelectedSession('Tile one')
    }

    cleanup()
    focus('workspace-group')
    renderSidebar('/reports', 'extension')
    expectOnlyCurrent('Reports')

    cleanup()
    disposeContributions()
    disposeContributions = noop
    focus('workspace-group')
    renderSidebar('/kanban', 'extension')
    expect(screen.queryByRole('button', { name: 'Kanban' })).toBeNull()
    expectOnlyCurrent(null)
    expectOnlySelectedSession(null)
  })

  it('keeps Home first-class across grouping and collapsed-session states', () => {
    const onNewSessionInWorkspace = vi.fn()
    const { container } = renderSidebar('/', 'chat', onNewSessionInWorkspace)

    expect(screen.getByRole('button', { name: 'Open Home' })).toBeTruthy()
    expect(container.querySelectorAll('[data-sidebar-home]')).toHaveLength(1)

    act(() => setSidebarRecentsOpen(false))
    expect(screen.getByRole('button', { name: 'Open Home' })).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: 'New session in Home' }))
    expect(onNewSessionInWorkspace).toHaveBeenCalledWith(null)

    fireEvent.click(screen.getByRole('button', { name: 'Open Home' }))
    expect($projectScope.get()).toBe('__no_project__')
    expect(container.querySelectorAll('[data-sidebar-home]')).toHaveLength(1)
  })

  it('keeps backend Home outside a project filter in project grouping', () => {
    act(() => {
      setSidebarGrouping('project')
      $sidebarProjectFilter.set(['p_project'])
    })

    const { container } = renderSidebar('/', 'chat')

    expect(screen.getByRole('button', { name: 'Open Home' })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Open Project One' })).toBeTruthy()
    expect(container.querySelectorAll('[data-sidebar-home="__no_project__"]')).toHaveLength(1)
  })

  it('synthesizes Home when an empty backend has no sessions and Recents is collapsed', () => {
    act(() => {
      $projectTree.set([])
      $sessions.set([])
      setSidebarRecentsOpen(false)
    })

    const { container } = renderSidebar('/', 'chat')

    expect(screen.getByRole('button', { name: 'Open Home' })).toBeTruthy()
    expect(container.querySelectorAll('[data-sidebar-home="__no_project__"]')).toHaveLength(1)
  })
})
