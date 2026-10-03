import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import type { atom } from 'nanostores'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { SidebarProjectTree } from '@/app/chat/sidebar/projects/workspace-groups'
import type { SessionInfo } from '@/hermes'
import type * as ProjectsStore from '@/store/projects'
import type { ProjectsReadOutcome } from '@/store/projects'
import type * as SessionDotStateStore from '@/store/session-dot-state'
import type { SessionDotState } from '@/store/session-dot-state'
import { deferred } from '@/test/deferred'

const projectsStore = vi.hoisted(() => ({
  fetchProjectSessions:
    vi.fn<(id: string, options?: { supersedable?: boolean }) => Promise<null | SidebarProjectTree>>(),
  refreshProjects: vi.fn(async (): Promise<ProjectsReadOutcome> => 'complete'),
  refreshProjectTree: vi.fn(async (): Promise<ProjectsReadOutcome> => 'complete')
}))

vi.mock('@/store/projects', async importOriginal => ({
  ...(await importOriginal<typeof ProjectsStore>()),
  ...projectsStore
}))

// The live-status map is derived from several session atoms; the cockpit only
// reads the result, so the test drives that result directly.
const dotStates = vi.hoisted(() => ({
  atom: null as unknown as ReturnType<typeof atom<Record<string, SessionDotState>>>
}))

vi.mock('@/store/session-dot-state', async importOriginal => {
  const actual = await importOriginal<typeof SessionDotStateStore>()
  const { atom: makeAtom } = await import('nanostores')
  dotStates.atom = makeAtom<Record<string, SessionDotState>>({})

  return { ...actual, $sessionDotStateById: dotStates.atom }
})

const {
  $projects,
  $projectsOwner,
  $projectsOwnerKey,
  $projectsReadStatus,
  $projectsRpcAvailable,
  $projectsRpcAvailableByOwner,
  $projectTree,
  $projectTreeOwner,
  $projectTreeReadStatus
} = await import('@/store/projects')

const { $activeGatewayProfile, setShowAllProfiles } = await import('@/store/profile')
const { $connection } = await import('@/store/session')
const { ProjectsView } = await import('.')

const session = (id: string, title: string, lastActive: number, extra: Partial<SessionInfo> = {}): SessionInfo =>
  ({ id, title, last_active: lastActive, started_at: lastActive, ...extra }) as unknown as SessionInfo

const atlas: SidebarProjectTree = {
  id: 'p_atlas',
  label: 'Atlas',
  path: '/work/atlas',
  sessionCount: 1,
  previewSessions: [session('s-plan', 'Plan the launch', 20)],
  repos: [
    {
      id: '/work/atlas',
      label: 'atlas',
      path: '/work/atlas',
      sessionCount: 1,
      groups: [
        { id: '/work/atlas::main', isHome: true, isMain: true, label: 'main', path: '/work/atlas', sessions: [] }
      ]
    }
  ]
}

function renderView(initialPath = '/projects') {
  return render(
    <MemoryRouter initialEntries={[initialPath]}>
      <ProjectsView />
    </MemoryRouter>
  )
}

// The connection identity Electron publishes; the cockpit only reads it.
const setConnection = (connectionId: string) => $connection.set({ connectionId } as never)

// As the real store does, a read publishes its verdict for the owner it was
// made for, and a read that landed tags the shared cache with that owner; the
// fixtures stand in for that owner's answer.
const listSettles = (outcome: 'complete' | 'failed') => async (): Promise<ProjectsReadOutcome> => {
  const owner = $projectsOwnerKey.get()

  if (outcome !== 'failed') {
    $projectsOwner.set(owner)
  }

  $projectsReadStatus.set({ outcome, owner })

  return outcome
}

const treeSettles = (outcome: 'complete' | 'failed') => async (): Promise<ProjectsReadOutcome> => {
  const owner = $projectsOwnerKey.get()

  if (outcome !== 'failed') {
    $projectTreeOwner.set(owner)
  }

  $projectTreeReadStatus.set({ outcome, owner })

  return outcome
}

beforeEach(() => {
  setConnection('local')
  projectsStore.fetchProjectSessions.mockResolvedValue(null)
  projectsStore.refreshProjects.mockImplementation(listSettles('complete'))
  projectsStore.refreshProjectTree.mockImplementation(treeSettles('complete'))
  $projectsRpcAvailable.set(true)
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
  $projectTree.set([])
  $projects.set([])
  $projectTreeOwner.set(null)
  $projectsOwner.set(null)
  $projectTreeReadStatus.set(null)
  $projectsReadStatus.set(null)
  $projectsRpcAvailable.set(null)
  $projectsRpcAvailableByOwner.set({})
  dotStates.atom.set({})
  $connection.set(null)
  $activeGatewayProfile.set('default')
  setShowAllProfiles(false)
})

const sessionsSectionOf = (detail: HTMLElement) =>
  within(detail).getByRole('heading', { name: 'Sessions' }).parentElement!

const hydratedAtlas = (sessions: SessionInfo[]): SidebarProjectTree => ({
  ...atlas,
  repos: atlas.repos.map(repo => ({ ...repo, groups: repo.groups.map(group => ({ ...group, sessions })) }))
})

describe('ProjectsView', () => {
  it('lists projects from the canonical tree and shows the selected project facts and sessions', async () => {
    $projectTree.set([atlas])

    renderView()

    fireEvent.click(await screen.findByRole('button', { name: /Atlas/ }))

    const detail = await screen.findByRole('region', { name: 'Atlas' })

    expect(within(detail).getAllByText('/work/atlas').length).toBeGreaterThan(0)
    expect(within(detail).getByText('main')).toBeTruthy()
    await waitFor(() => expect(within(detail).getByText('Plan the launch')).toBeTruthy())
  })

  it('shows sessions the backend assigned to the project and labels only the ones actually running', async () => {
    // A session moved into the project by hand is in the backend lanes but not
    // in the tree's three-row preview.
    projectsStore.fetchProjectSessions.mockResolvedValue(
      hydratedAtlas([session('s-plan', 'Plan the launch', 20), session('s-moved', 'Moved here by hand', 10)])
    )
    dotStates.atom.set({ 's-moved': 'working', 's-plan': 'idle' })
    $projectTree.set([atlas])

    renderView('/projects?project=p_atlas')

    const detail = await screen.findByRole('region', { name: 'Atlas' })
    const activeSection = within(detail).getByRole('heading', { name: 'Active now' }).parentElement!
    const sessionsSection = within(detail).getByRole('heading', { name: 'Sessions' }).parentElement!

    await waitFor(() => expect(within(activeSection).getByText('Moved here by hand')).toBeTruthy())
    expect(within(activeSection).getByText('Working')).toBeTruthy()
    expect(within(sessionsSection).getByText('Plan the launch')).toBeTruthy()
    // Idle is not a status: nothing is invented for a session with no live turn.
    expect(within(sessionsSection).queryByText('Idle')).toBeNull()
    expect(projectsStore.fetchProjectSessions).toHaveBeenCalledWith('p_atlas', { supersedable: false })
  })

  it('links to the global Artifacts surface without offering an unscoped Kanban destination', async () => {
    $projectTree.set([atlas])
    renderView('/projects?project=p_atlas')

    const detail = await screen.findByRole('region', { name: 'Atlas' })

    expect(within(detail).getByRole('button', { name: /Artifacts/ })).toBeTruthy()
    expect(within(detail).queryByRole('button', { name: /Kanban/ })).toBeNull()
  })

  it('keeps the empty cockpit read-only: no project creation, only a refresh', async () => {
    renderView()

    expect(await screen.findByText('No projects yet')).toBeTruthy()
    expect(screen.queryByRole('button', { name: /New project/ })).toBeNull()
    expect(screen.getAllByRole('button').map(button => button.textContent?.trim())).toEqual(['Refresh projects'])
  })

  it('limits the populated cockpit to Artifacts, sidebar scoping, session opening, search and refresh', async () => {
    projectsStore.fetchProjectSessions.mockResolvedValue(hydratedAtlas([session('s-plan', 'Plan the launch', 20)]))
    $projectTree.set([atlas])
    renderView('/projects?project=p_atlas')

    const detail = await screen.findByRole('region', { name: 'Atlas' })
    await waitFor(() => expect(within(detail).getByText('Plan the launch')).toBeTruthy())

    const labels = screen.getAllByRole('button').map(button => button.getAttribute('aria-label') || button.textContent)

    expect(labels).toEqual([
      'Refresh projects',
      expect.stringContaining('Atlas'),
      'Artifacts',
      'Show in sidebar',
      expect.stringContaining('Plan the launch')
    ])
    expect(screen.queryByRole('button', { name: /Kanban|New project|Rename|Delete|Archive/ })).toBeNull()
  })

  it.each<[string, Record<'list' | 'tree', 'complete' | 'failed'>]>([
    ['the tree read', { list: 'complete', tree: 'failed' }],
    ['both reads', { list: 'failed', tree: 'failed' }],
    ['the list read with no tree yet', { list: 'failed', tree: 'complete' }]
  ])('ends a failed first load of %s in an error state with a retry that recovers in place', async (_label, ok) => {
    // A stale backend probe never answered: availability stays unknown.
    $projectsRpcAvailable.set(null)
    projectsStore.refreshProjects.mockImplementationOnce(listSettles(ok.list))
    projectsStore.refreshProjectTree.mockImplementationOnce(treeSettles(ok.tree))

    renderView()

    expect(await screen.findByText("Couldn't load projects")).toBeTruthy()
    expect(screen.queryByText('No projects yet')).toBeNull()
    expect(screen.queryByText('Loading projects')).toBeNull()

    projectsStore.refreshProjectTree.mockImplementationOnce(async () => {
      $projectTree.set([atlas])

      return treeSettles('complete')()
    })
    fireEvent.click(screen.getByRole('button', { name: 'Retry' }))

    expect(await screen.findByRole('button', { name: /Atlas/ })).toBeTruthy()
    expect(screen.queryByText("Couldn't load projects")).toBeNull()
  })

  it('keeps cached projects visible with a notice when only the list read fails', async () => {
    projectsStore.refreshProjects.mockImplementationOnce(listSettles('failed'))
    $projectTree.set([atlas])

    renderView()

    expect(await screen.findByText(/Couldn't refresh every project detail/)).toBeTruthy()
    expect(screen.getByRole('button', { name: /Atlas/ })).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: 'Refresh projects' }))

    await waitFor(() => expect(screen.queryByText(/Couldn't refresh every project detail/)).toBeNull())
  })

  it('Refresh retries a failed session read and replaces a stale title at unchanged count and time', async () => {
    projectsStore.fetchProjectSessions.mockRejectedValueOnce(new Error('gateway read failed'))
    $projectTree.set([atlas])
    renderView('/projects?project=p_atlas')

    const detail = await screen.findByRole('region', { name: 'Atlas' })

    expect(await within(detail).findByText(/Couldn't load every session/)).toBeTruthy()

    projectsStore.fetchProjectSessions.mockResolvedValueOnce(hydratedAtlas([session('s-plan', 'Old title', 20)]))
    fireEvent.click(screen.getByRole('button', { name: 'Refresh projects' }))

    await waitFor(() => expect(within(sessionsSectionOf(detail)).getByText('Old title')).toBeTruthy())
    expect(within(detail).queryByText(/Couldn't load every session/)).toBeNull()

    // Same id, count and last-active: only the backend title changed.
    projectsStore.fetchProjectSessions.mockResolvedValueOnce(
      hydratedAtlas([session('s-plan', 'Renamed on the backend', 20)])
    )
    fireEvent.keyDown(window, { key: 'r' })

    await waitFor(() => expect(within(sessionsSectionOf(detail)).getByText('Renamed on the backend')).toBeTruthy())
    expect(within(detail).queryByText('Old title')).toBeNull()
    expect(projectsStore.fetchProjectSessions).toHaveBeenCalledTimes(3)
  })

  it('never claims no agents are working when hydration failed and a live session is outside the preview', async () => {
    const preview = [
      session('s-idle-1', 'Idle preview one', 30),
      session('s-idle-2', 'Idle preview two', 29),
      session('s-idle-3', 'Idle preview three', 28)
    ]

    $projectTree.set([{ ...atlas, sessionCount: 5, previewSessions: preview }])
    dotStates.atom.set({ 's-idle-1': 'idle', 's-idle-2': 'idle', 's-idle-3': 'idle', 's-live': 'working' })
    projectsStore.fetchProjectSessions.mockRejectedValueOnce(new Error('gateway read failed'))
    renderView('/projects?project=p_atlas')

    const detail = await screen.findByRole('region', { name: 'Atlas' })
    const activeSection = () => within(detail).getByRole('heading', { name: 'Active now' }).parentElement!

    expect(await within(detail).findByText(/Couldn't load every session/)).toBeTruthy()
    expect(within(activeSection()).getByText(/Can't confirm which agents are working/)).toBeTruthy()
    expect(within(detail).queryByText(/No agents are working/)).toBeNull()

    projectsStore.fetchProjectSessions.mockResolvedValueOnce(
      hydratedAtlas([...preview, session('s-live', 'Live outside the preview', 10), session('s-old', 'Old chat', 5)])
    )
    fireEvent.click(screen.getByRole('button', { name: 'Refresh projects' }))

    await waitFor(() => expect(within(activeSection()).getByText('Live outside the preview')).toBeTruthy())
    expect(within(activeSection()).getByText('Working')).toBeTruthy()
    expect(within(detail).queryByText(/Can't confirm which agents are working/)).toBeNull()
  })

  it('never reports a null session answer as a complete hydration', async () => {
    // `fetchProjectSessions` answers null when its owner moved mid-read or the
    // project is gone: no answer, not "these are all the sessions".
    projectsStore.fetchProjectSessions.mockResolvedValue(null)
    $projectTree.set([atlas])
    renderView('/projects?project=p_atlas')

    const detail = await screen.findByRole('region', { name: 'Atlas' })

    expect(await within(detail).findByText(/Couldn't load every session/)).toBeTruthy()
    expect(within(detail).getByText('Plan the launch')).toBeTruthy()
  })

  it("never paints another owner's sessions across A → B → A with the same project id and late answers", async () => {
    $activeGatewayProfile.set('alpha')
    $projectTree.set([atlas])

    const alphaFirst = deferred<null | SidebarProjectTree>()
    const beta = deferred<null | SidebarProjectTree>()
    const alphaAgain = deferred<null | SidebarProjectTree>()
    projectsStore.fetchProjectSessions
      .mockReturnValueOnce(alphaFirst.promise)
      .mockReturnValueOnce(beta.promise)
      .mockReturnValueOnce(alphaAgain.promise)

    renderView('/projects?project=p_atlas')
    await screen.findByRole('region', { name: 'Atlas' })

    // On B (whose tree also holds a `p_atlas`), B's own sessions are correct to show.
    act(() => $activeGatewayProfile.set('beta'))
    await waitFor(() => expect(projectsStore.fetchProjectSessions).toHaveBeenCalledTimes(2))
    await act(async () => beta.resolve(hydratedAtlas([session('s-beta', 'Beta private plan', 99)])))
    expect(within(await screen.findByRole('region', { name: 'Atlas' })).getByText('Beta private plan')).toBeTruthy()

    // Back on A, with A's read still in flight: B's rows are gone at once.
    act(() => $activeGatewayProfile.set('alpha'))
    expect(screen.queryByText('Beta private plan')).toBeNull()
    await waitFor(() => expect(projectsStore.fetchProjectSessions).toHaveBeenCalledTimes(3))
    const detail = await screen.findByRole('region', { name: 'Atlas' })
    expect(within(detail).queryByText('Beta private plan')).toBeNull()

    // A's departed first read lands late and is dropped.
    await act(async () => alphaFirst.resolve(hydratedAtlas([session('s-alpha-old', 'Alpha stale read', 50)])))
    expect(screen.queryByText('Alpha stale read')).toBeNull()

    // A's current read fails: A's preview plus the notice — never B's rows.
    await act(async () => alphaAgain.reject(new Error('gateway read failed')))

    expect(await within(detail).findByText(/Couldn't load every session/)).toBeTruthy()
    expect(screen.queryByText('Beta private plan')).toBeNull()
    expect(within(detail).getByText('Plan the launch')).toBeTruthy()
  })

  it('drops hydrated sessions when the connection changes under the same profile and project id', async () => {
    projectsStore.fetchProjectSessions
      .mockResolvedValueOnce(hydratedAtlas([session('s-local', 'Local machine chat', 40)]))
      .mockReturnValueOnce(new Promise(() => undefined))
    $projectTree.set([atlas])
    renderView('/projects?project=p_atlas')

    const detail = await screen.findByRole('region', { name: 'Atlas' })
    await waitFor(() => expect(within(detail).getByText('Local machine chat')).toBeTruthy())

    act(() => setConnection('remote-box'))

    expect(screen.queryByText('Local machine chat')).toBeNull()
    // The remote tree also holds a `p_atlas`; only its own preview may stand in.
    const remoteDetail = await screen.findByRole('region', { name: 'Atlas' })
    expect(within(remoteDetail).queryByText('Local machine chat')).toBeNull()
    expect(within(remoteDetail).getByText('Plan the launch')).toBeTruthy()
  })

  it("states the All Profiles limitation instead of passing a preview off as the project's sessions", async () => {
    setShowAllProfiles(true)
    // Five sessions across two profiles; the tree only previews three, and the
    // live one is outside the preview.
    $projectTree.set([
      {
        ...atlas,
        sessionCount: 5,
        previewSessions: [
          session('s-a1', 'Alpha preview one', 30, { profile: 'alpha' }),
          session('s-b1', 'Beta preview one', 29, { profile: 'beta' }),
          session('s-a2', 'Alpha preview two', 28, { profile: 'alpha' })
        ]
      }
    ])
    dotStates.atom.set({ 's-live': 'working', 's-b1': 'working' })

    renderView('/projects?project=p_atlas')
    const detail = await screen.findByRole('region', { name: 'Atlas' })

    expect(within(detail).getByText(/All Profiles can't list one project's sessions/)).toBeTruthy()
    expect(within(detail).getByText('5 sessions')).toBeTruthy()
    // No row that could open through the wrong owner, and no claim of being complete.
    expect(within(detail).queryByText(/preview/)).toBeNull()
    expect(within(detail).queryByText('Working')).toBeNull()
    expect(within(detail).queryByRole('heading', { name: 'Active now' })).toBeNull()
    expect(within(detail).getByRole('button', { name: 'Show in sidebar' })).toBeTruthy()
    expect(projectsStore.fetchProjectSessions).not.toHaveBeenCalled()
    // `projects.list` answers for one profile only; All Profiles reads just the tree.
    expect(projectsStore.refreshProjects).not.toHaveBeenCalled()
    expect(projectsStore.refreshProjectTree).toHaveBeenCalled()
  })

  it('exposes full paths for truncated primary, repository, lane and folder paths', async () => {
    const longRoot = '/Users/someone/Development/clients/very-long-organisation-name/monorepo-with-a-long-name'
    const worktree = `${longRoot}/.worktrees/feature-with-an-exceptionally-descriptive-branch-name`
    $projectTree.set([
      {
        ...atlas,
        path: longRoot,
        repos: [
          {
            id: longRoot,
            label: 'monorepo-with-a-long-name',
            path: longRoot,
            sessionCount: 1,
            groups: [
              { id: `${longRoot}::main`, isHome: true, isMain: true, label: 'main', path: longRoot, sessions: [] },
              {
                id: `${worktree}::wt`,
                label: 'feature-with-an-exceptionally-descriptive-branch-name',
                path: worktree,
                sessions: []
              }
            ]
          }
        ]
      }
    ])

    renderView('/projects?project=p_atlas')
    const detail = await screen.findByRole('region', { name: 'Atlas' })

    for (const element of within(detail).getAllByText(longRoot)) {
      expect(element.getAttribute('title')).toBe(longRoot)
    }

    expect(within(detail).getByText(worktree).getAttribute('title')).toBe(worktree)
    expect(
      within(detail).getByText('feature-with-an-exceptionally-descriptive-branch-name').getAttribute('title')
    ).toBe('feature-with-an-exceptionally-descriptive-branch-name')
  })

  it('explains an older backend instead of spinning forever', async () => {
    $projectsRpcAvailableByOwner.set({ [$projectsOwnerKey.get()]: false })

    renderView()

    expect(await screen.findByText('Projects are unavailable')).toBeTruthy()
  })

  it('never takes focus from the transcript when it mounts or the tree refreshes in the background', async () => {
    const composer = globalThis.document.createElement('textarea')
    globalThis.document.body.append(composer)
    composer.focus()

    try {
      $projectTree.set([atlas])
      renderView('/projects?project=p_atlas')
      await screen.findByRole('region', { name: 'Atlas' })

      act(() => $projectTree.set([{ ...atlas, sessionCount: 2, lastActive: 30 }]))
      await waitFor(() => expect(projectsStore.fetchProjectSessions).toHaveBeenCalledTimes(2))

      expect(globalThis.document.activeElement).toBe(composer)
    } finally {
      composer.remove()
    }
  })
})
