import { beforeEach, describe, expect, it, vi } from 'vitest'

import { $sidebarAgentsGrouped } from '@/store/layout'

import {
  $activeProjectId,
  $projectScope,
  $projectsRpcAvailable,
  $projectTree,
  $worktreeRefreshToken,
  ALL_PROJECTS,
  createProject,
  ensureProjectForFolder,
  enterProject,
  exitProjectScope,
  openProjectCreate,
  pickProjectFolder,
  refreshProjects,
  refreshWorktrees
} from './projects'

vi.mock('@/i18n', () => ({
  translateNow: (key: string) => key
}))

vi.mock('@/store/notifications', () => ({
  notify: vi.fn()
}))

vi.mock('@/lib/desktop-fs', () => ({
  desktopDefaultCwd: vi.fn(),
  isDesktopFsRemoteMode: vi.fn(),
  selectDesktopPaths: vi.fn(),
  writeDesktopFileText: vi.fn()
}))

vi.mock('@/store/gateway', () => ({
  activeGateway: vi.fn(),
  ensureActiveGatewayOpen: vi.fn()
}))

const fs = await import('@/lib/desktop-fs')
const desktopDefaultCwd = vi.mocked(fs.desktopDefaultCwd)
const isDesktopFsRemoteMode = vi.mocked(fs.isDesktopFsRemoteMode)
const selectDesktopPaths = vi.mocked(fs.selectDesktopPaths)

const gw = await import('@/store/gateway')
const activeGateway = vi.mocked(gw.activeGateway)
const notifications = await import('@/store/notifications')
const notify = vi.mocked(notifications.notify)

describe('project scope', () => {
  beforeEach(() => {
    window.localStorage.clear()
    $projectScope.set(ALL_PROJECTS)
  })

  it('defaults to ALL_PROJECTS', () => {
    expect($projectScope.get()).toBe(ALL_PROJECTS)
  })

  it('enterProject scopes the sidebar to the project id', () => {
    // setActiveProject fires best-effort (no gateway in test → it rejects and is
    // swallowed); the synchronous scope change is what matters here.
    enterProject('p_123')
    expect($projectScope.get()).toBe('p_123')
  })

  it('exitProjectScope returns to the overview', () => {
    enterProject('p_123')
    exitProjectScope()
    expect($projectScope.get()).toBe(ALL_PROJECTS)
  })

  it('entering the synthetic No-project bucket still scopes (no active pin)', () => {
    enterProject('__no_project__')
    expect($projectScope.get()).toBe('__no_project__')
  })

  it('persists the scope to localStorage', () => {
    enterProject('p_abc')
    expect(window.localStorage.getItem('hermes.desktop.projectScope')).toBe('p_abc')
  })
})

describe('worktree refresh', () => {
  it('refreshWorktrees bumps the probe token so useRepoWorktreeMap refetches', () => {
    const before = $worktreeRefreshToken.get()
    refreshWorktrees()
    expect($worktreeRefreshToken.get()).toBe(before + 1)
  })
})

describe('pickProjectFolder', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('uses the remote-aware directory picker locally', async () => {
    isDesktopFsRemoteMode.mockReturnValue(false)
    selectDesktopPaths.mockResolvedValue(['/local/repo'])

    await expect(pickProjectFolder()).resolves.toBe('/local/repo')
    expect(selectDesktopPaths).toHaveBeenCalledWith({ defaultPath: undefined, directories: true, multiple: false })
  })

  it('seeds the picker with the backend cwd on a remote gateway', async () => {
    isDesktopFsRemoteMode.mockReturnValue(true)
    desktopDefaultCwd.mockResolvedValue({ branch: 'main', cwd: '/backend/work' })
    selectDesktopPaths.mockResolvedValue(['/backend/work/repo'])

    await expect(pickProjectFolder()).resolves.toBe('/backend/work/repo')
    expect(selectDesktopPaths).toHaveBeenCalledWith({
      defaultPath: '/backend/work',
      directories: true,
      multiple: false
    })
  })

  it('returns null when the picker is cancelled (empty selection)', async () => {
    isDesktopFsRemoteMode.mockReturnValue(false)
    selectDesktopPaths.mockResolvedValue([])

    await expect(pickProjectFolder()).resolves.toBeNull()
  })
})

describe('createProject', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    $sidebarAgentsGrouped.set(false)
    $activeProjectId.set(null)
    $projectsRpcAvailable.set(null)
  })

  it('creates the project and flips into the grouped view so a blank slate shows it', async () => {
    const created = { folders: [], id: 'p_new', name: 'Demo', primary_path: '/srv/demo' }

    const request = vi.fn(async (method: string) => {
      if (method === 'projects.create') {
        return { project: created }
      }

      // Reconcile (fire-and-forget) re-reads list + tree; echo the project back
      // so the optimistic state survives instead of being wiped to empty.
      return { active_id: 'p_new', projects: [created], scoped_session_ids: [] }
    })

    activeGateway.mockReturnValue({ connectionState: 'open', request } as never)

    const result = await createProject({ folders: ['/srv/demo'], name: 'Demo', use: true })

    expect(result).toEqual(created)
    expect(request).toHaveBeenCalledWith('projects.create', expect.objectContaining({ name: 'Demo' }))
    expect($sidebarAgentsGrouped.get()).toBe(true)
    expect($activeProjectId.get()).toBe('p_new')
  })

  it('marks the backend stale and surfaces a friendly error when projects.create is missing', async () => {
    activeGateway.mockReturnValue({
      connectionState: 'open',
      request: vi.fn().mockRejectedValue(new Error('unknown method: projects.create'))
    } as never)

    await expect(createProject({ folders: ['/srv/demo'], name: 'Demo' })).rejects.toThrow(
      'sidebar.projects.staleBackend'
    )
    expect($projectsRpcAvailable.get()).toBe(false)
  })
  it('creates an explicit Project named for an unowned folder before entering it', async () => {
    const created = { folders: [{ path: '/srv/new-project' }], id: 'p_new', name: 'new-project', primary_path: '/srv/new-project' }

    const request = vi.fn(async (method: string) => {
      if (method === 'projects.create') {
        return { project: created }
      }

      if (method === 'projects.tree') {
        return { active_id: null, projects: [], scoped_session_ids: [] }
      }

      return { active_id: 'p_new', projects: [created], scoped_session_ids: [] }
    })

    $projectTree.set([])
    activeGateway.mockReturnValue({ connectionState: 'open', request } as never)

    $sidebarAgentsGrouped.set(false)

    await expect(ensureProjectForFolder('/srv/new-project')).resolves.toBe('/srv/new-project')

    expect(request).toHaveBeenCalledWith(
      'projects.create',
      expect.objectContaining({ folders: ['/srv/new-project'], name: 'new-project', primary_path: '/srv/new-project', use: false })
    )
    expect($projectScope.get()).toBe('p_new')
    expect($sidebarAgentsGrouped.get()).toBe(true)
  })

  it('enters the deepest Project that already owns the folder without creating another', async () => {
    const owners = [
      {
        color: null,
        icon: null,
        id: 'p_parent',
        isAuto: false,
        label: 'parent',
        path: '/srv',
        previewSessions: [],
        repos: [],
        sessionCount: 0
      },
      {
        color: null,
        icon: null,
        id: 'p_child',
        isAuto: false,
        label: 'child',
        path: '/srv/team/child',
        previewSessions: [],
        repos: [],
        sessionCount: 0
      }
    ]

    const request = vi.fn(async (method: string) => {
      if (method === 'projects.tree') {
        return { active_id: null, projects: owners, scoped_session_ids: [] }
      }

      return {}
    })

    $sidebarAgentsGrouped.set(false)
    $projectTree.set([])
    activeGateway.mockReturnValue({ connectionState: 'open', request } as never)

    await expect(ensureProjectForFolder('/srv/team/child/src')).resolves.toBe('/srv/team/child/src')

    expect($projectScope.get()).toBe('p_child')
    expect($sidebarAgentsGrouped.get()).toBe(true)
    expect(request).not.toHaveBeenCalledWith('projects.create', expect.anything())
  })

  it('reuses an owning Project for Windows-style paths', async () => {
    const owner = {
      color: null,
      icon: null,
      id: 'p_windows',
      isAuto: false,
      label: 'windows',
      path: 'C:\\Users\\dev\\project',
      previewSessions: [],
      repos: [],
      sessionCount: 0
    }

    const request = vi.fn(async (method: string) => {
      if (method === 'projects.tree') {
        return { active_id: null, projects: [owner], scoped_session_ids: [] }
      }

      return {}
    })

    $projectTree.set([])
    activeGateway.mockReturnValue({ connectionState: 'open', request } as never)

    await expect(ensureProjectForFolder('C:\\Users\\dev\\project\\src')).resolves.toBe(
      'C:\\Users\\dev\\project\\src'
    )

    expect($projectScope.get()).toBe('p_windows')
    expect(request).not.toHaveBeenCalledWith('projects.create', expect.anything())
  })

  it('rejects without leaking scope when the backend creates no Project', async () => {
    const request = vi.fn(async (method: string) => {
      if (method === 'projects.tree') {
        return { active_id: null, projects: [], scoped_session_ids: [] }
      }

      if (method === 'projects.create') {
        return { project: null }
      }

      return {}
    })

    $projectScope.set(ALL_PROJECTS)
    $projectTree.set([])
    activeGateway.mockReturnValue({ connectionState: 'open', request } as never)

    await expect(ensureProjectForFolder('/srv/orphan')).rejects.toThrow('Could not create a Project')
    expect($projectScope.get()).toBe(ALL_PROJECTS)
  })

  it('cancels (returns null, no Project) when a session becomes active mid-resolution', async () => {
    const request = vi.fn(async (method: string) => {
      if (method === 'projects.tree') {
        return { active_id: null, projects: [], scoped_session_ids: [] }
      }

      return {}
    })

    $projectScope.set(ALL_PROJECTS)
    $projectTree.set([])
    activeGateway.mockReturnValue({ connectionState: 'open', request } as never)

    await expect(ensureProjectForFolder('/srv/late', () => true)).resolves.toBeNull()
    expect(request).not.toHaveBeenCalledWith('projects.create', expect.anything())
    expect($projectScope.get()).toBe(ALL_PROJECTS)
  })

  it('surfaces the error and does not create when the authoritative tree read fails', async () => {
    const request = vi.fn(async (method: string) => {
      if (method === 'projects.tree') {
        throw new Error('tree unavailable')
      }

      return {}
    })

    $projectScope.set(ALL_PROJECTS)
    $projectTree.set([])
    activeGateway.mockReturnValue({ connectionState: 'open', request } as never)

    await expect(ensureProjectForFolder('/srv/maybe-owned')).rejects.toThrow('tree unavailable')
    expect(request).not.toHaveBeenCalledWith('projects.create', expect.anything())
    expect($projectScope.get()).toBe(ALL_PROJECTS)
  })

  it('does not switch scope when a session starts during the create RPC', async () => {
    const created = { folders: [{ path: '/srv/race' }], id: 'p_race', name: 'race', primary_path: '/srv/race' }
    let cancelled = false

    const request = vi.fn(async (method: string) => {
      if (method === 'projects.tree') {
        return { active_id: null, projects: [], scoped_session_ids: [] }
      }

      if (method === 'projects.create') {
        cancelled = true

        return { project: created }
      }

      return {}
    })

    $projectScope.set(ALL_PROJECTS)
    $projectTree.set([])
    activeGateway.mockReturnValue({ connectionState: 'open', request } as never)

    await expect(ensureProjectForFolder('/srv/race', () => cancelled)).resolves.toBeNull()
    expect(request).toHaveBeenCalledWith('projects.create', expect.anything())
    expect($projectScope.get()).toBe(ALL_PROJECTS)
    expect($activeProjectId.get()).toBeNull()
    expect($sidebarAgentsGrouped.get()).toBe(false)
  })
})


describe('projects RPC capability', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    $projectsRpcAvailable.set(null)
  })

  it('marks the backend stale when projects.list is missing', async () => {
    activeGateway.mockReturnValue({
      connectionState: 'open',
      request: vi.fn().mockRejectedValue(new Error('unknown method: projects.list'))
    } as never)

    await refreshProjects()

    expect($projectsRpcAvailable.get()).toBe(false)
  })

  it('blocks opening the create dialog once the backend is known stale', () => {
    $projectsRpcAvailable.set(false)

    openProjectCreate()

    expect(notify).toHaveBeenCalledWith(
      expect.objectContaining({ kind: 'warning', message: 'sidebar.projects.staleBackend' })
    )
  })
})
