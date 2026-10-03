import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import type { atom } from 'nanostores'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { SidebarProjectTree } from '@/app/chat/sidebar/projects/workspace-groups'
import { registry } from '@/contrib/registry'
import type { SessionInfo } from '@/hermes'
import type * as ProjectsStore from '@/store/projects'
import type * as SessionDotStateStore from '@/store/session-dot-state'
import type { SessionDotState } from '@/store/session-dot-state'

import { ROUTES_AREA } from '../routes'

const projectsStore = vi.hoisted(() => ({
  fetchProjectSessions:
    vi.fn<(id: string, options?: { supersedable?: boolean }) => Promise<null | SidebarProjectTree>>(),
  refreshProjects: vi.fn(async () => undefined),
  refreshProjectTree: vi.fn(async () => undefined)
}))

vi.mock('@/store/projects', async importOriginal => ({
  ...(await importOriginal<typeof ProjectsStore>()),
  ...projectsStore
}))

// The live-status map is derived from several session atoms; the cockpit only
// reads the result, so the test drives that result directly.
const dotStates = vi.hoisted(() => ({ atom: null as unknown as ReturnType<typeof atom<Record<string, SessionDotState>>> }))

vi.mock('@/store/session-dot-state', async importOriginal => {
  const actual = await importOriginal<typeof SessionDotStateStore>()
  const { atom: makeAtom } = await import('nanostores')
  dotStates.atom = makeAtom<Record<string, SessionDotState>>({})

  return { ...actual, $sessionDotStateById: dotStates.atom }
})

const { $projects, $projectsRpcAvailable, $projectTree } = await import('@/store/projects')
const { ProjectsView } = await import('.')

const session = (id: string, title: string, lastActive: number): SessionInfo =>
  ({ id, title, last_active: lastActive, started_at: lastActive }) as unknown as SessionInfo

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
      groups: [{ id: '/work/atlas::main', isHome: true, isMain: true, label: 'main', path: '/work/atlas', sessions: [] }]
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

beforeEach(() => {
  projectsStore.fetchProjectSessions.mockResolvedValue(null)
  $projectsRpcAvailable.set(true)
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
  $projectTree.set([])
  $projects.set([])
  $projectsRpcAvailable.set(null)
  dotStates.atom.set({})
})

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

  it('links to Artifacts always and to Kanban only while its route is contributed', async () => {
    $projectTree.set([atlas])
    renderView('/projects?project=p_atlas')

    const detail = await screen.findByRole('region', { name: 'Atlas' })

    expect(within(detail).getByRole('button', { name: /Artifacts/ })).toBeTruthy()
    expect(within(detail).queryByRole('button', { name: /Kanban/ })).toBeNull()

    const dispose = registry.register({
      area: ROUTES_AREA,
      data: { path: '/kanban' },
      id: 'kanban-test',
      render: () => null
    })

    try {
      await waitFor(() => expect(within(detail).getByRole('button', { name: /Kanban/ })).toBeTruthy())
    } finally {
      act(() => dispose())
    }
  })

  it('offers project creation when there are no projects', async () => {
    renderView()

    expect(await screen.findByText('No projects yet')).toBeTruthy()
    expect(screen.getByRole('button', { name: 'New project' })).toBeTruthy()
  })

  it('explains an older backend instead of spinning forever', async () => {
    $projectsRpcAvailable.set(false)

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
