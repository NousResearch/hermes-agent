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

const {
  $projects,
  $projectsOwner,
  $projectsOwnerKey,
  $projectsRpcAvailable,
  $projectsRpcAvailableByOwner,
  $projectTree,
  $projectTreeOwner,
  $projectTreeReadStatus,
  $projectsReadStatus,
  deleteProject,
  refreshProjects,
  refreshProjectTree,
  updateProject
} = await import('@/store/projects')

const { ProjectsView } = await import('.')

type Respond = (method: string, params: Record<string, unknown>) => Promise<unknown>

interface FakeGateway {
  connectionState: 'open'
  request: ReturnType<typeof vi.fn<Respond>>
}

function openGateway(respond: Respond): FakeGateway {
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
  $projectTreeOwner.set(null)
  $projectsOwner.set(null)
  $projectTreeReadStatus.set(null)
  $projectsReadStatus.set(null)
  $projectsRpcAvailable.set(null)
  $projectsRpcAvailableByOwner.set({})
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

    openGateway(
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
    openGateway(() => Promise.reject(new Error('remote read failed')))
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
    const gateway = openGateway(ownerResponder({ alpha: answersWith(alpha) }))
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

    openGateway(method => {
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
    openGateway(() => Promise.reject(new Error('All Profiles reads no single gateway')))
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

// Unqualified descriptors (no registry `connectionId`) as Electron publishes
// them for an env/settings primary: only the endpoint tells them apart.
const unqualified = (host: string) =>
  ({
    baseUrl: `https://${host}:9119`,
    mode: 'remote',
    remoteHost: host,
    token: `${host}-secret-token`,
    wsUrl: `wss://${host}:9119/api/ws?token=${host}-secret-token`
  }) as never

const hydratedFor = (fixture: OwnerFixture, title: string) => ({
  project: {
    ...fixture.tree,
    repos: [
      {
        groups: [
          {
            id: `${fixture.tree.path}::main`,
            isMain: true,
            label: 'main',
            path: fixture.tree.path,
            sessions: [session(`s-${title}`, title, 30)]
          }
        ],
        id: fixture.tree.path,
        label: 'atlas',
        path: fixture.tree.path,
        sessionCount: 1
      }
    ]
  }
})

describe('ProjectsView owner isolation across unqualified connections (real store)', () => {
  it('keeps two backends with no registry id apart across delayed, failed and successful A → B → A reads', async () => {
    const alphaHost = unqualified('alpha.example.test')
    const betaHost = unqualified('beta.example.test')
    const alphaText = [...ownerText(alpha), 'Alpha hydrated chat']
    const betaText = [...ownerText(beta), 'Beta hydrated chat']

    const expectNone = (texts: string[]) => {
      for (const text of texts) {
        expect(screen.queryAllByText(text, { exact: false })).toEqual([])
      }
    }

    const serve = (fixture: OwnerFixture, hydrated: string, gate?: Promise<unknown>) =>
      openGateway(async method => {
        await gate

        if (method === 'projects.project_sessions') {
          return hydratedFor(fixture, hydrated)
        }

        return answersWith(fixture)(method)
      })

    $activeGatewayProfile.set('default')
    $connection.set(alphaHost)
    serve(alpha, 'Alpha hydrated chat')
    renderView()

    const alphaDetail = await screen.findByRole('region', { name: 'Alpha Atlas' })
    await waitFor(() => expect(within(alphaDetail).getByText('Alpha hydrated chat')).toBeTruthy())

    // A → B: B's reads are delayed, then fail.
    const betaFailure = deferred()
    openGateway(() => betaFailure.promise.then(() => Promise.reject(new Error('beta unreachable'))))
    act(() => $connection.set(betaHost))
    expectNone(alphaText)
    expect(await screen.findByRole('status', { name: 'Loading projects' })).toBeTruthy()
    expectNone(alphaText)

    await act(async () => betaFailure.resolve())
    expect(await screen.findByText("Couldn't load projects")).toBeTruthy()
    expectNone(alphaText)

    // B recovers on Retry — only B's rows.
    serve(beta, 'Beta hydrated chat')
    fireEvent.click(screen.getByRole('button', { name: 'Retry' }))
    const betaDetail = await screen.findByRole('region', { name: 'Beta Atlas' })
    await waitFor(() => expect(within(betaDetail).getByText('Beta hydrated chat')).toBeTruthy())
    expectNone(alphaText)

    // B → A with A's reads delayed: B's rows leave at once.
    const alphaGate = deferred()
    serve(alpha, 'Alpha hydrated chat', alphaGate.promise)
    act(() => $connection.set(alphaHost))
    expectNone(betaText)

    await act(async () => alphaGate.resolve())
    const alphaAgain = await screen.findByRole('region', { name: 'Alpha Atlas' })
    await waitFor(() => expect(within(alphaAgain).getByText('Alpha hydrated chat')).toBeTruthy())
    expectNone(betaText)

    // The owner identity names the endpoint, never its credentials.
    expect($projectsOwnerKey.get()).not.toMatch(/secret|token/)
  })
})

describe("ProjectsView never shows a departed owner's late write (real store)", () => {
  it.each([
    ['a rejected update rolls back', 'update', 'reject'],
    ['a delete response lands', 'delete', 'resolve'],
    ['a rejected delete rolls back', 'delete', 'reject']
  ] as const)('keeps B on B when A’s %s after B loaded', async (_label, write, settle) => {
    const alphaWrite = deferred<unknown>()

    openGateway((method, params) => {
      if (method === 'projects.project_sessions') {
        // B's hydration fails too: only the tree's preview could stand in.
        return Promise.reject(new Error('hydration failed'))
      }

      if (params.profile === 'alpha') {
        return method === 'projects.update' || method === 'projects.delete'
          ? alphaWrite.promise
          : answersWith(alpha)(method)
      }

      return answersWith(beta)(method)
    })

    renderView()
    await screen.findByRole('region', { name: 'Alpha Atlas' })
    await waitFor(() => expect(screen.getByText('Alpha description')).toBeTruthy())

    // A's optimistic write paints at once; its RPC stays in flight.
    const pendingWrite =
      write === 'update' ? updateProject('p_atlas', { name: 'Alpha renamed' }) : deleteProject('p_atlas')

    const writeSettled = pendingWrite.catch(() => undefined)
    await waitFor(() => expect(gatewayStore.activeGateway).toHaveBeenCalled())

    act(() => $activeGatewayProfile.set('beta'))
    const betaDetail = await screen.findByRole('region', { name: 'Beta Atlas' })
    await waitFor(() => expect(within(betaDetail).getByText('Beta description')).toBeTruthy())

    await act(async () => {
      if (settle === 'reject') {
        alphaWrite.reject(new Error('alpha write failed'))
      } else {
        alphaWrite.resolve(listPayload(alpha))
      }

      await writeSettled
    })

    expect(await screen.findByRole('region', { name: 'Beta Atlas' })).toBeTruthy()
    expect(screen.getByText('Beta description')).toBeTruthy()
    expectNoOwnerText(alpha)
    expect(screen.queryByText('Alpha renamed')).toBeNull()
  })
})

describe('ProjectsView follows background tree reads for its owner (real store)', () => {
  const profileError = { error: 'database is locked', profile: 'beta' }
  const fanOut = () => vi.mocked(hermes.hermesApi)
  const background = (read: () => Promise<unknown>) => act(async () => void (await read()))

  it('warns on background partial and total failures, keeps the last good list, and clears on recovery', async () => {
    openGateway(() => Promise.reject(new Error('All Profiles reads no single gateway')))
    setShowAllProfiles(true)
    fanOut().mockResolvedValueOnce({ ...treePayload(alpha), errors: [] })
    renderView('/projects')

    expect(await screen.findByRole('button', { name: /Alpha Atlas/ })).toBeTruthy()
    await waitFor(() =>
      expect(screen.getByRole('button', { name: 'Refresh projects' })).toHaveProperty('disabled', false)
    )

    // A background sync (not the cockpit) reads a partial fan-out.
    fanOut().mockResolvedValueOnce({ ...treePayload(alpha), errors: [profileError] })
    await background(refreshProjectTree)
    expect(await screen.findByText(/Some profiles couldn't be read/)).toBeTruthy()

    // Then every profile fails: last good list stays, never "No projects yet".
    fanOut().mockResolvedValueOnce({ active_id: null, errors: [profileError], projects: [], scoped_session_ids: [] })
    await background(refreshProjectTree)
    expect(await screen.findByText(/Couldn't refresh every project detail/)).toBeTruthy()
    expect(screen.getByRole('button', { name: /Alpha Atlas/ })).toBeTruthy()
    expect(screen.queryByText('No projects yet')).toBeNull()

    // A healthy background read clears every warning on its own.
    fanOut().mockResolvedValueOnce({ ...treePayload(alpha), errors: [] })
    await background(refreshProjectTree)
    await waitFor(() => expect(screen.queryByText(/Couldn't refresh every project detail/)).toBeNull())
    expect(screen.queryByText(/Some profiles couldn't be read/)).toBeNull()
    expect(screen.getByRole('button', { name: /Alpha Atlas/ })).toBeTruthy()
  })

  it("ignores a departed owner's background outcome", async () => {
    openGateway(ownerResponder({ alpha: answersWith(alpha) }))
    setShowAllProfiles(true)
    fanOut().mockResolvedValueOnce({ ...treePayload(alpha), errors: [] })
    renderView('/projects')
    await screen.findByRole('button', { name: /Alpha Atlas/ })

    // An All Profiles background read is still in flight when the user picks
    // a single profile; it then lands with errors.
    const departed = deferred<unknown>()
    fanOut().mockReturnValueOnce(departed.promise as never)
    const pending = refreshProjectTree()
    act(() => setShowAllProfiles(false))
    await waitFor(() => expect(gatewayStore.activeGateway).toHaveBeenCalled())
    expect(await screen.findByRole('button', { name: /Alpha Atlas/ })).toBeTruthy()

    await act(async () => {
      departed.resolve({ active_id: null, errors: [profileError], projects: [], scoped_session_ids: [] })
      await pending
    })

    expect(screen.queryByText(/Couldn't refresh every project detail/)).toBeNull()
    expect(screen.queryByText(/Some profiles couldn't be read/)).toBeNull()
    expect(screen.queryByText("Couldn't load projects")).toBeNull()
    expect(screen.getByRole('button', { name: /Alpha Atlas/ })).toBeTruthy()
  })
})

describe('ProjectsView scopes backend capability to its owner (real store)', () => {
  it("offers B a retry after A's missing project methods, and recovers B in place", async () => {
    let betaReads: (method: string) => Promise<unknown> = () => Promise.reject(new Error('gateway read failed'))

    openGateway(
      ownerResponder({
        alpha: method => Promise.reject(new Error(`unknown method: ${method}`)),
        beta: method => betaReads(method)
      })
    )

    renderView()
    expect(await screen.findByText('Projects are unavailable')).toBeTruthy()

    // B supports projects but its first reads fail transiently.
    act(() => $activeGatewayProfile.set('beta'))
    expect(await screen.findByText("Couldn't load projects")).toBeTruthy()
    expect(screen.queryByText('Projects are unavailable')).toBeNull()

    betaReads = answersWith(beta)
    fireEvent.click(screen.getByRole('button', { name: 'Retry' }))
    expect(await screen.findByRole('region', { name: 'Beta Atlas' })).toBeTruthy()

    // A's own evidence still stands for A.
    act(() => $activeGatewayProfile.set('alpha'))
    expect(await screen.findByText('Projects are unavailable')).toBeTruthy()
    expectNoOwnerText(beta)
  })
})
