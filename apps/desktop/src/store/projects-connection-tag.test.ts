import { atom } from 'nanostores'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { SidebarProjectTree } from '@/app/chat/sidebar/projects/workspace-groups'
import { hermesApi } from '@/hermes'
import { $gateway, activeGateway, activeGatewayConnectionId, isActivePrimary } from '@/store/gateway'
import { $activeGatewayProfile, $profiles, setShowAllProfiles } from '@/store/profile'
import { $sessions } from '@/store/session'
import { sessionOwnerRouteFromRow } from '@/store/session-request-router'
import { deferred } from '@/test/deferred'

import { $projectTree, fetchProjectSessions, refreshProjectTree } from './projects'

vi.mock('@/store/gateway', () => ({
  $gateway: atom(null),
  activeGateway: vi.fn(),
  activeGatewayConnectionId: vi.fn(),
  isActivePrimary: vi.fn(),
  ensureActiveGatewayOpen: vi.fn()
}))
vi.mock('@/i18n', () => ({ translateNow: (key: string) => key }))
vi.mock('@/hermes', () => ({
  getHermesConfig: vi.fn(),
  getProfiles: vi.fn(),
  hermesApi: vi.fn(),
  setApiRequestProfile: vi.fn(),
  STARTUP_REQUEST_TIMEOUT_MS: 1000
}))

function treeProject(): SidebarProjectTree {
  // Neither row is in the recents store; opening it must use tree provenance.
  const row = { id: 'tree-only', profile: 'default', title: 'Outside recents' }

  return {
    id: 'project',
    label: 'Project',
    path: '/project',
    sessionCount: 1,
    previewSessions: [row as never],
    repos: [
      {
        id: 'repo',
        label: 'Repo',
        path: '/project',
        sessionCount: 1,
        groups: [{ id: 'lane', label: 'main', path: '/project', sessions: [row as never] }]
      }
    ]
  }
}

// Default topology: Home is the window primary, so `local` is a secondary.
function selectConnection(connectionId: string | null, primary = connectionId !== 'local') {
  vi.mocked(activeGatewayConnectionId).mockReturnValue(connectionId)
  vi.mocked(isActivePrimary).mockReturnValue(primary)
}

function rows(project: SidebarProjectTree) {
  return [
    ...(project.previewSessions ?? []),
    ...project.repos.flatMap(repo => repo.groups.flatMap(group => group.sessions))
  ]
}

function connect(request = vi.fn()) {
  const gateway = { connectionState: 'open', request }
  vi.mocked(activeGateway).mockReturnValue(gateway as never)
  $gateway.set(gateway as never)

  return request
}

beforeEach(() => {
  vi.clearAllMocks()
  $activeGatewayProfile.set('default')
  // A real all-profiles window: the hidden single-profile preference never
  // counts as the effective scope (see projects.test.ts).
  $profiles.set([{ is_default: true, name: 'default' } as never, { is_default: false, name: 'coder' } as never])
  setShowAllProfiles(false)
  $sessions.set([])
  $projectTree.set([])
  selectConnection('home')
})

describe('project session connection provenance', () => {
  it('tags tree-only overview and hydrated rows across Home → local → Home', async () => {
    for (const connectionId of ['home', 'local', 'home']) {
      const project = treeProject()
      connect(vi.fn().mockResolvedValue({ projects: [project], project, scoped_session_ids: [] }))
      selectConnection(connectionId)

      await refreshProjectTree()
      const hydrated = await fetchProjectSessions(project.id)

      expect($sessions.get()).toEqual([])
      expect(hydrated).not.toBeNull()

      for (const row of [...rows($projectTree.get()[0]), ...rows(hydrated!)]) {
        expect(sessionOwnerRouteFromRow(row)).toEqual({ connectionId, profile: row.profile })
      }

      // Stamp renderer copies, not the response object shared by transport caches.
      expect(rows(project).every(row => row.connection_id === undefined)).toBe(true)
      const before = $projectTree.get()
      await refreshProjectTree()
      expect($projectTree.get()).toBe(before)
    }
  })

  it.each(['overview', 'hydrated', 'all-profiles'])(
    'drops a late %s response when only the connection identity changes',
    async surface => {
      const pending = deferred<unknown>()
      const started = deferred<void>()

      const request = vi.fn(() => {
        started.resolve()

        return pending.promise
      })

      connect(request)
      vi.mocked(hermesApi).mockImplementation(request as never)
      setShowAllProfiles(surface === 'all-profiles')

      const result = surface === 'hydrated' ? fetchProjectSessions('project') : refreshProjectTree()
      await started.promise
      // Same profile and socket object: the registry identity itself must be guarded.
      selectConnection('local')
      pending.resolve({ projects: [treeProject()], project: treeProject(), scoped_session_ids: [] })
      const value = await result

      expect($projectTree.get()).toEqual([])

      if (surface === 'hydrated') {
        expect(value).toBeNull()
      }
    }
  )

  it('tags the all-profiles tree without replacing each row profile', async () => {
    const project = treeProject()
    project.previewSessions![0].profile = 'coder'
    vi.mocked(hermesApi).mockResolvedValue({ projects: [project], scoped_session_ids: [] })
    selectConnection('local')
    setShowAllProfiles(true)

    await refreshProjectTree()

    for (const row of rows($projectTree.get()[0])) {
      expect(sessionOwnerRouteFromRow(row)).toEqual({ connectionId: 'local', profile: 'coder' })
    }
  })

  it('leaves This device rows bare on the primary path (legacy profile door)', async () => {
    const project = treeProject()
    connect(vi.fn().mockResolvedValue({ projects: [project], project, scoped_session_ids: [] }))
    selectConnection('local', true)

    await refreshProjectTree()
    const hydrated = await fetchProjectSessions(project.id)

    for (const row of [...rows($projectTree.get()[0]), ...rows(hydrated!)]) {
      expect(row).not.toHaveProperty('connection_id')
      expect(sessionOwnerRouteFromRow(row)).toBeUndefined()
    }
  })

  it.each(['home', 'local'])('never re-owns a row that already names its connection (%s active)', async active => {
    const project = treeProject()

    for (const row of rows(project)) {
      row.connection_id = 'elsewhere'
    }

    vi.mocked(hermesApi).mockResolvedValue({ projects: [project], scoped_session_ids: [] })
    selectConnection(active)
    setShowAllProfiles(true)

    await refreshProjectTree()

    for (const row of rows($projectTree.get()[0])) {
      expect(sessionOwnerRouteFromRow(row)).toEqual({ connectionId: 'elsewhere', profile: 'default' })
    }
  })

  it('leaves legacy rows untagged when no registry identity exists', async () => {
    const project = treeProject()
    connect(vi.fn().mockResolvedValue({ projects: [project], project, scoped_session_ids: [] }))
    selectConnection(null)

    await refreshProjectTree()
    const hydrated = await fetchProjectSessions(project.id)

    for (const row of [...rows($projectTree.get()[0]), ...rows(hydrated!)]) {
      expect(row).not.toHaveProperty('connection_id')
      expect(sessionOwnerRouteFromRow(row)).toBeUndefined()
    }
  })
})
