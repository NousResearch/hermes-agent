import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { SidebarProjectTree } from '@/app/chat/sidebar/projects/workspace-groups'
import type * as Hermes from '@/hermes'
import type { SessionInfo } from '@/hermes'
import type * as GatewayStore from '@/store/gateway'
import { deferred } from '@/test/deferred'
import type { ProjectInfo } from '@/types/hermes'

// The cockpit over the REAL projects store: only the transport is faked, so
// owner tags, supersession and payload handling are exercised end to end.
vi.mock('@/store/gateway', async importOriginal => ({
  ...(await importOriginal<typeof GatewayStore>()),
  activeGateway: vi.fn(),
  ensureActiveGatewayOpen: vi.fn()
}))

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<typeof Hermes>()),
  hermesApi: vi.fn()
}))

const gatewayStore = await import('@/store/gateway')
const hermes = await import('@/hermes')
const { $activeGatewayProfile, setShowAllProfiles } = await import('@/store/profile')
const { $projectScope, ALL_PROJECTS } = await import('@/store/project-scope')
const { $sidebarAgentsGrouped, setSidebarAgentsGrouped } = await import('@/store/layout')
const { $connection } = await import('@/store/session')

const { $projects, $projectsRpcAvailable, $projectTree, refreshProjects, refreshProjectTree } =
  await import('@/store/projects')

const { ProjectsView } = await import('.')

type Respond = (method: string, params: Record<string, unknown>) => Promise<unknown>

interface FakeGateway {
  connectionState: 'open'
  request: ReturnType<typeof vi.fn<Respond>>
}

function useGateway(respond: Respond): FakeGateway {
  const gateway: FakeGateway = { connectionState: 'open', request: vi.fn<Respond>(respond) }
  vi.mocked(gatewayStore.activeGateway).mockReturnValue(gateway as never)
  vi.mocked(gatewayStore.ensureActiveGatewayOpen).mockResolvedValue(gateway as never)

  return gateway
}

const setConnection = (connectionId: string) => $connection.set({ connectionId } as never)

const session = (id: string, title: string, lastActive: number): SessionInfo =>
  ({ id, title, last_active: lastActive, started_at: lastActive }) as unknown as SessionInfo

// Two owners that both have a project `p_atlas`: same id, nothing else shared.
interface OwnerFixture {
  info: ProjectInfo
  tree: SidebarProjectTree
}

function owner(name: string): OwnerFixture {
  const path = `/${name.toLowerCase()}/atlas`

  return {
    info: {
      archived: false,
      board_slug: null,
      color: null,
      created_at: 0,
      description: `${name} description`,
      folders: [],
      icon: null,
      id: 'p_atlas',
      name: `${name} Atlas`,
      primary_path: path,
      slug: 'atlas'
    },
    tree: {
      id: 'p_atlas',
      label: `${name} Atlas`,
      path,
      sessionCount: 1,
      previewSessions: [session(`s-${name}`, `${name} preview chat`, 20)],
      repos: []
    }
  }
}

const alpha = owner('Alpha')
const beta = owner('Beta')

const listPayload = (fixture: OwnerFixture) => ({ active_id: null, projects: [fixture.info] })
const treePayload = (fixture: OwnerFixture) => ({ active_id: null, projects: [fixture.tree], scoped_session_ids: [] })

/** Answers list/tree reads from `byProfile`; session hydration always fails so
 *  the tree's preview is what the detail falls back to. */
function ownerResponder(byProfile: Record<string, (method: string) => Promise<unknown>>): Respond {
  return (method, params) => {
    if (method === 'projects.project_sessions') {
      return Promise.reject(new Error('hydration failed'))
    }

    const answer = byProfile[String(params.profile)]

    return answer ? answer(method) : Promise.reject(new Error(`no fixture for ${String(params.profile)}`))
  }
}

const answersWith = (fixture: OwnerFixture) => (method: string) =>
  Promise.resolve(method === 'projects.list' ? listPayload(fixture) : treePayload(fixture))

function renderView(initialPath = '/projects?project=p_atlas') {
  return render(
    <MemoryRouter initialEntries={[initialPath]}>
      <ProjectsView />
    </MemoryRouter>
  )
}

const ownerText = (fixture: OwnerFixture) => [
  fixture.tree.label,
  fixture.tree.path!,
  fixture.info.description!,
  fixture.tree.previewSessions![0]!.title!
]

function expectNoOwnerText(fixture: OwnerFixture) {
  for (const text of ownerText(fixture)) {
    expect(screen.queryAllByText(text, { exact: false })).toEqual([])
  }
}

beforeEach(() => {
  setConnection('local')
  $activeGatewayProfile.set('alpha')
  setShowAllProfiles(false)
  $projectsRpcAvailable.set(true)
})

afterEach(() => {
  cleanup()
  $projectScope.set(ALL_PROJECTS)
  setSidebarAgentsGrouped(false)
  vi.clearAllMocks()
  $projectTree.set([])
  $projects.set([])
  $projectsRpcAvailable.set(null)
  setShowAllProfiles(false)
  $activeGatewayProfile.set('default')
  $connection.set(null)
})

describe('ProjectsView owner isolation (real store)', () => {
  it("never shows a departed owner's project, details or preview sessions across A → B → A and a connection change", async () => {
    // B's first reads stay in flight until the test fails them.
    const betaFailure = deferred()

    let betaReads = (_method: string): Promise<unknown> =>
      betaFailure.promise.then(() => Promise.reject(new Error('beta read failed')))

    let alphaReads = answersWith(alpha)

    useGateway(
      ownerResponder({
        alpha: method => alphaReads(method),
        beta: method => betaReads(method)
      })
    )

    renderView()

    const detail = await screen.findByRole('region', { name: 'Alpha Atlas' })
    await waitFor(() => expect(within(detail).getByText('Alpha preview chat')).toBeTruthy())
    expect(within(detail).getByText('Alpha description')).toBeTruthy()

    // A → B while B's reads are still in flight: nothing of A remains.
    act(() => $activeGatewayProfile.set('beta'))
    expectNoOwnerText(alpha)
    expect(await screen.findByRole('status', { name: 'Loading projects' })).toBeTruthy()
    expectNoOwnerText(alpha)

    // B's reads fail: an honest failure with a retry, still nothing of A.
    await act(async () => betaFailure.resolve())

    expect(await screen.findByText("Couldn't load projects")).toBeTruthy()
    expectNoOwnerText(alpha)

    // Retry recovers B — and only B.
    betaReads = answersWith(beta)
    fireEvent.click(screen.getByRole('button', { name: 'Retry' }))

    const betaDetail = await screen.findByRole('region', { name: 'Beta Atlas' })
    await waitFor(() => expect(within(betaDetail).getByText('Beta preview chat')).toBeTruthy())
    expect(within(betaDetail).getByText('Beta description')).toBeTruthy()
    expectNoOwnerText(alpha)

    // B → A while A's reads are delayed: B's rows are gone at once.
    const alphaPending = deferred()

    alphaReads = async method => {
      await alphaPending.promise

      return answersWith(alpha)(method)
    }

    act(() => $activeGatewayProfile.set('alpha'))
    expectNoOwnerText(beta)

    await act(async () => alphaPending.resolve())
    const alphaAgain = await screen.findByRole('region', { name: 'Alpha Atlas' })
    await waitFor(() => expect(within(alphaAgain).getByText('Alpha preview chat')).toBeTruthy())
    expectNoOwnerText(beta)

    // A connection change under the same profile name is a different owner:
    // the new machine's failing reads must not fall back to this one's data.
    useGateway(() => Promise.reject(new Error('remote read failed')))
    act(() => setConnection('remote-box'))
    expectNoOwnerText(alpha)
    expect(await screen.findByText("Couldn't load projects")).toBeTruthy()
    expectNoOwnerText(alpha)
  })
})

const PROJECT_READS = new Set(['projects.list', 'projects.project_sessions', 'projects.tree'])

const settle = () => act(() => new Promise<void>(resolve => setTimeout(resolve, 0)))

describe('ProjectsView "Show in sidebar" (real store)', () => {
  it.each([
    ['a single profile', false],
    ['All Profiles', true]
  ])('only scopes the sidebar in %s — the durable active project is never written', async (_label, allProfiles) => {
    const gateway = useGateway(ownerResponder({ alpha: answersWith(alpha) }))
    vi.mocked(hermes.hermesApi).mockResolvedValue({ ...treePayload(alpha), errors: [] })
    setShowAllProfiles(allProfiles)

    renderView()

    const detail = await screen.findByRole('region', { name: 'Alpha Atlas' })
    fireEvent.click(within(detail).getByRole('button', { name: 'Show in sidebar' }))
    await settle()

    expect($projectScope.get()).toBe('p_atlas')
    expect($sidebarAgentsGrouped.get()).toBe(true)
    const writes = gateway.request.mock.calls.map(([method]) => method).filter(method => !PROJECT_READS.has(method))
    expect(writes).toEqual([])
  })
})

describe('ProjectsView read supersession (real store)', () => {
  it('follows a chain of same-owner background reads to their failure instead of claiming an empty success', async () => {
    // Every list/tree read stays in flight until the test settles it.
    const pending: Record<string, ReturnType<typeof deferred<unknown>>[]> = {
      'projects.list': [],
      'projects.tree': []
    }

    useGateway(method => {
      const read = pending[method]

      if (!read) {
        return Promise.reject(new Error('hydration failed'))
      }

      const next = deferred<unknown>()
      read.push(next)

      return next.promise
    })

    renderView()
    await waitFor(() => expect(pending['projects.tree']).toHaveLength(1))
    await waitFor(() => expect(pending['projects.list']).toHaveLength(1))

    // The sidebar refreshes twice for the same owner (fire-and-forget), each
    // superseding the read before it.
    void refreshProjects()
    void refreshProjectTree()
    await waitFor(() => expect(pending['projects.tree']).toHaveLength(2))
    void refreshProjects()
    void refreshProjectTree()
    await waitFor(() => expect(pending['projects.tree']).toHaveLength(3))
    expect(pending['projects.list']).toHaveLength(3)

    const empty = { list: listPayload(alpha), tree: { active_id: null, projects: [], scoped_session_ids: [] } }
    empty.list.projects = []

    // The newest reads fail; the ones they superseded then answer "nothing".
    await act(async () => {
      pending['projects.list']![2]!.reject(new Error('list read failed'))
      pending['projects.tree']![2]!.reject(new Error('tree read failed'))
    })
    await act(async () => {
      for (const index of [1, 0]) {
        pending['projects.list']![index]!.resolve(empty.list)
        pending['projects.tree']![index]!.resolve(empty.tree)
      }
    })

    expect(await screen.findByText("Couldn't load projects")).toBeTruthy()
    expect(screen.queryByText('No projects yet')).toBeNull()

    // A healthy retry recovers in place.
    fireEvent.click(screen.getByRole('button', { name: 'Retry' }))
    await waitFor(() => expect(pending['projects.tree']).toHaveLength(4))
    await waitFor(() => expect(pending['projects.list']).toHaveLength(4))
    await act(async () => {
      pending['projects.list']![3]!.resolve(listPayload(alpha))
      pending['projects.tree']![3]!.resolve(treePayload(alpha))
    })

    expect(await screen.findByRole('region', { name: 'Alpha Atlas' })).toBeTruthy()
    expect(screen.queryByText("Couldn't load projects")).toBeNull()
  })
})

describe('ProjectsView All Profiles payload errors (real store)', () => {
  const profileError = { error: 'database is locked', profile: 'beta' }
  const healthy = { ...treePayload(alpha), errors: [] }
  const fanOut = () => vi.mocked(hermes.hermesApi)

  beforeEach(() => {
    useGateway(() => Promise.reject(new Error('All Profiles reads no single gateway')))
    setShowAllProfiles(true)
  })

  it('shows partial projects as explicitly incomplete until a healthy read clears it', async () => {
    fanOut().mockResolvedValueOnce({ ...treePayload(alpha), errors: [profileError] })

    renderView('/projects')

    expect(await screen.findByRole('button', { name: /Alpha Atlas/ })).toBeTruthy()
    expect(await screen.findByText(/Some profiles couldn't be read/)).toBeTruthy()

    fanOut().mockResolvedValueOnce(healthy)
    fireEvent.click(screen.getByRole('button', { name: 'Refresh projects' }))

    await waitFor(() => expect(screen.queryByText(/Some profiles couldn't be read/)).toBeNull())
    expect(screen.getByRole('button', { name: /Alpha Atlas/ })).toBeTruthy()
  })

  it('never turns errors with no projects into an empty cockpit or a wiped last-good list', async () => {
    fanOut().mockResolvedValueOnce({ active_id: null, errors: [profileError], projects: [], scoped_session_ids: [] })

    renderView('/projects')

    expect(await screen.findByText("Couldn't load projects")).toBeTruthy()
    expect(screen.queryByText('No projects yet')).toBeNull()

    fanOut().mockResolvedValueOnce(healthy)
    fireEvent.click(screen.getByRole('button', { name: 'Retry' }))
    expect(await screen.findByRole('button', { name: /Alpha Atlas/ })).toBeTruthy()

    // Every profile fails on the next read: the last good list stays, flagged.
    fanOut().mockResolvedValueOnce({ active_id: null, errors: [profileError], projects: [], scoped_session_ids: [] })
    fireEvent.click(screen.getByRole('button', { name: 'Refresh projects' }))

    expect(await screen.findByText(/Couldn't refresh every project detail/)).toBeTruthy()
    expect(screen.getByRole('button', { name: /Alpha Atlas/ })).toBeTruthy()
    expect($projectTree.get().map(project => project.id)).toEqual(['p_atlas'])

    fanOut().mockResolvedValueOnce(healthy)
    fireEvent.click(screen.getByRole('button', { name: 'Refresh projects' }))

    await waitFor(() => expect(screen.queryByText(/Couldn't refresh every project detail/)).toBeNull())
    expect(screen.getByRole('button', { name: /Alpha Atlas/ })).toBeTruthy()
  })
})
