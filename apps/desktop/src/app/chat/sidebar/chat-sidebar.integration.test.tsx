// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen, within } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { group, split } from '@/components/pane-shell/tree/model'
import { $layoutTree, noteActiveTreeGroup } from '@/components/pane-shell/tree/store'
import { SidebarProvider } from '@/components/ui/sidebar'
import { registry } from '@/contrib/registry'
import { $connectionsRegistry } from '@/store/connection-registry-state'
import { $sidebarMessagingOpenIds, setSidebarAgentsGrouped, setSidebarGrouping } from '@/store/layout'
import { $activeGatewayProfile, $profiles, setShowAllProfiles } from '@/store/profile'
import { $projectScope, ALL_PROJECTS } from '@/store/project-scope'
import { $projectTree } from '@/store/projects'
import {
  $currentCwd,
  $messagingSessions,
  $messagingTruncated,
  $selectedStoredSessionId,
  $sessions,
  $sessionsLoadError,
  $sessionsLoading,
  $workspaceCwdOwner
} from '@/store/session'
import { $removedSessionIds } from '@/store/session-removal'
import { SIDEBAR_NAV_PREFS_AREA } from '@/store/sidebar-nav'
import { makeSessionInfo } from '@/test/session-info'

import { type AppView, ROUTES_AREA, SIDEBAR_NAV_AREA } from '../../routes'

import { $gatewayGroupCollapsed } from './gateway-group-preferences'

import { ChatSidebar } from './index'

const noop = () => {}

const noopAsync = async () => {}

const sessionRows = [
  makeSessionInfo({ id: 'tile-one', last_active: 2, profile: 'default', started_at: 1, title: 'Tile one' }),
  makeSessionInfo({ id: 'tile-two', last_active: 2, profile: 'default', started_at: 1, title: 'Tile two' })
]

const renderSidebar = (pathname: string, currentView: AppView, onRetrySessions: () => Promise<void> = noopAsync) =>
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
          onNewSessionInWorkspace={noop}
          onNewSessionSplit={noop}
          onResumeSession={noop}
          onRetrySessions={onRetrySessions}
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
      ['/capabilities', 'capabilities', 'Capabilities'],
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

  // Teardown proof: the loader disposes a plugin's contributions on disable,
  // and that disposer alone must bring the row back — no store to clear.
  it('hides a nav row while a sidebarNav.prefs contribution is registered and restores it on dispose', () => {
    renderSidebar('/kanban', 'extension')
    expect(screen.getByRole('button', { name: 'Kanban' })).toBeTruthy()

    let dispose = () => {}

    act(() => {
      dispose = registry.register({ area: SIDEBAR_NAV_PREFS_AREA, id: 'prefs', data: { hide: ['kanban-nav'] } })
    })
    expect(screen.queryByRole('button', { name: 'Kanban' })).toBeNull()
    // A hidden row is a preference, not a removal: the sibling nav row stays.
    expect(screen.getByRole('button', { name: 'Reports' })).toBeTruthy()

    act(() => dispose())
    expect(screen.getByRole('button', { name: 'Kanban' })).toBeTruthy()
  })
})

// #67600: a cold-start read that failed used to render as an empty account.
describe('ChatSidebar cold-start load failure', () => {
  afterEach(() => {
    cleanup()
    $sessions.set([])
    $sessionsLoading.set(true)
    $sessionsLoadError.set(false)
  })

  it('offers retry in place of the empty state', () => {
    const retry = vi.fn(noopAsync)
    $sessions.set([])
    $sessionsLoading.set(false)
    $sessionsLoadError.set(true)

    renderSidebar('/', 'chat', retry)

    expect(screen.getByText('Could not load sessions')).toBeTruthy()
    expect(screen.queryByText('No sessions yet')).toBeNull()

    fireEvent.click(screen.getByRole('button', { name: 'Retry' }))
    expect(retry).toHaveBeenCalledTimes(1)
  })
})

// Entering a project is a scope switch: the conversation main is showing keeps
// its workspace, so Files/Review and the composer's Git context can't drift to
// the project while the transcript stays on the old chat (#72772).
describe('ChatSidebar project entry', () => {
  const project = {
    id: '/repos/new-project',
    label: 'new-project',
    path: '/repos/new-project',
    repos: [],
    sessionCount: 0
  }

  beforeEach(() => {
    setSidebarAgentsGrouped(true)
    $projectTree.set([project])
    $currentCwd.set('/repos/old-project')
  })

  afterEach(() => {
    cleanup()
    $projectScope.set(ALL_PROJECTS)
    $projectTree.set([])
    setSidebarAgentsGrouped(false)
    $currentCwd.set('')
    $selectedStoredSessionId.set(null)
    $workspaceCwdOwner.set(null)
    $sessions.set([])
  })

  it("leaves a stored conversation's workspace alone", () => {
    $sessions.set(sessionRows)
    $selectedStoredSessionId.set('tile-one')
    $workspaceCwdOwner.set('tile-one')
    $projectScope.set(project.id)

    renderSidebar('/tile-one', 'chat')

    expect($currentCwd.get()).toBe('/repos/old-project')
    expect($workspaceCwdOwner.get()).toBe('tile-one')
  })

  it('re-homes a fresh draft into the entered project', () => {
    $projectScope.set(project.id)

    renderSidebar('/', 'chat')

    expect($currentCwd.get()).toBe(project.path)
  })
})

// Messaging platforms group rows by owner the same way recents does once every
// profile is on screen, so a Telegram thread is attributable to its profile
// (#87715). The platform's row cap and load-more stay the section's.
describe('ChatSidebar section labels', () => {
  const realProject = {
    id: '/repos/new-project',
    label: 'new-project',
    path: '/repos/new-project',
    repos: [],
    sessionCount: 0
  }

  // The synthetic Home bucket the backend emits when no project claimed a
  // session (no cwd / deleted workspace): one lane, no repo structure.
  const homeOnly = [
    {
      ...realProject,
      id: '__no_project__',
      label: 'Home',
      path: null,
      isNoProject: true,
      sessionCount: 2
    }
  ]

  afterEach(() => {
    cleanup()
    $projectScope.set(ALL_PROJECTS)
    $projectTree.set([])
    setSidebarAgentsGrouped(false)
    $sessions.set([])
  })

  it('labels the section "Projects" when the grouped tree has a real project', () => {
    setSidebarAgentsGrouped(true)
    $projectTree.set([realProject])

    renderSidebar('/', 'chat')

    expect(screen.getByText('Projects')).toBeTruthy()
    expect(screen.queryByText('Sessions')).toBeNull()
  })

  it('keeps the "Sessions" label when grouping is on but no projects exist', () => {
    // The source-filter toggle with an empty tree: the section lists plain
    // chat sessions, so heading it "Projects" mislabeled them (#62537).
    setSidebarAgentsGrouped(true)
    $projectTree.set([])
    $sessions.set(sessionRows)

    renderSidebar('/', 'chat')

    expect(screen.getByText('Sessions')).toBeTruthy()
    expect(screen.queryByText('Projects')).toBeNull()
  })

  it('keeps the "Sessions" label when the grouped tree is only the synthetic Home bucket', () => {
    // No real projects — every session fell into Home. A lone Home bucket is
    // the flat session list wearing a project costume, not a switcher.
    setSidebarAgentsGrouped(true)
    $projectTree.set(homeOnly)
    $sessions.set(sessionRows)

    renderSidebar('/', 'chat')

    expect(screen.getByText('Sessions')).toBeTruthy()
    expect(screen.queryByText('Projects')).toBeNull()
  })
})

describe('ChatSidebar messaging owners', () => {
  const telegram = (id: string, profile: string, last_active: number) =>
    makeSessionInfo({ connection_id: 'local', id, last_active, profile, source: 'telegram', title: id })

  // Newest first: the 3-row cap lets default-1, work-1 and default-2 through.
  const threads = [
    telegram('default-1', 'default', 50),
    telegram('work-1', 'work', 40),
    telegram('default-2', 'default', 30),
    telegram('work-2', 'work', 20),
    telegram('default-3', 'default', 10)
  ]

  let root: HTMLElement

  const mount = () => {
    root = renderSidebar('/', 'chat').container
  }

  const telegramGroups = () =>
    [...root.querySelectorAll<HTMLElement>('[data-gateway-group]')].filter(node =>
      node.dataset.gatewayGroup!.startsWith(JSON.stringify(['messaging:telegram']).slice(0, -1))
    )

  const titlesIn = (node: HTMLElement) =>
    threads.map(thread => thread.id).filter(title => within(node).queryByText(title))

  beforeEach(() => {
    $connectionsRegistry.set({
      version: 2,
      primary: 'local',
      secureTokenStorage: true,
      connections: [{ id: 'local', label: 'This computer', kind: 'local', tokenSet: false, tokenPreview: null }]
    } as NonNullable<typeof $connectionsRegistry.value>)
    $profiles.set([
      { name: 'default', is_default: true },
      { name: 'work', is_default: false }
    ] as typeof $profiles.value)
    $sessions.set([
      makeSessionInfo({ connection_id: 'local', id: 'desk', last_active: 60, profile: 'default', title: 'desk' })
    ])
    $messagingSessions.set(threads)
    $messagingTruncated.set(false)
    $sidebarMessagingOpenIds.set(['telegram'])
  })

  afterEach(() => {
    cleanup()
    setSidebarGrouping('date')
    setShowAllProfiles(false)
    $activeGatewayProfile.set('default')
    $gatewayGroupCollapsed.set([])
    $sidebarMessagingOpenIds.set([])
    $messagingSessions.set([])
    $sessions.set([])
    $profiles.set([])
    $connectionsRegistry.set(null)
  })

  it('groups the capped rows by owner, pages the platform as a whole, and keeps its own collapse keys', () => {
    setSidebarGrouping('profile')
    mount()

    const [defaultGroup, workGroup] = telegramGroups()

    expect(telegramGroups()).toHaveLength(2)
    expect(titlesIn(defaultGroup)).toEqual(['default-1', 'default-2'])
    expect(titlesIn(workGroup)).toEqual(['work-1'])

    fireEvent.click(screen.getByRole('button', { name: 'Load 2 more' }))

    expect(titlesIn(telegramGroups()[0])).toEqual(['default-1', 'default-2', 'default-3'])
    expect(titlesIn(telegramGroups()[1])).toEqual(['work-1', 'work-2'])
    expect(screen.queryByRole('button', { name: /^Load \d+ more$/ })).toBeNull()

    fireEvent.click(within(telegramGroups()[0]).getByRole('button', { name: 'Hide default sessions' }))

    expect(screen.queryByText('default-1')).toBeNull()
    expect(screen.getByText('desk')).toBeTruthy()
  })

  it('tags rows with their profile under other groupings and stays flat when scoped to one profile', () => {
    setShowAllProfiles(true)
    setSidebarGrouping('date')
    mount()

    const row = (title: string) => screen.getByText(title).closest('.group.row-hover') as HTMLElement

    expect(telegramGroups()).toHaveLength(0)
    expect(within(row('work-1')).getByRole('img', { name: 'Profile: work' })).toBeTruthy()

    cleanup()
    setSidebarGrouping('profile')
    setShowAllProfiles(false)
    mount()

    expect(telegramGroups()).toHaveLength(0)
    expect(screen.queryByText('work-1')).toBeNull()
    expect(within(row('default-1')).queryByRole('img', { name: /^Profile:/ })).toBeNull()
  })
})
