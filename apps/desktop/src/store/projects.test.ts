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
  enterProject,
  exitProjectScope,
  openProjectCreate,
  pickProjectFolder,
  refreshProjects,
  refreshWorktrees,
  scanAndRecordRepos
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

vi.mock('@/lib/desktop-git', () => ({
  desktopGit: vi.fn()
}))

vi.mock('@/store/gateway', () => ({
  activeGateway: vi.fn(),
  ensureActiveGatewayOpen: vi.fn()
}))

const fs = await import('@/lib/desktop-fs')
const desktopDefaultCwd = vi.mocked(fs.desktopDefaultCwd)
const isDesktopFsRemoteMode = vi.mocked(fs.isDesktopFsRemoteMode)
const selectDesktopPaths = vi.mocked(fs.selectDesktopPaths)

const git = await import('@/lib/desktop-git')
const desktopGit = vi.mocked(git.desktopGit)

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

describe('repo discovery scan', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('records scan results on the gateway that was active when the scan started', async () => {
    let finishScan: (repos: Array<{ label: string; root: string }>) => void = () => undefined

    const scanRepos = vi.fn(
      () =>
        new Promise<Array<{ label: string; root: string }>>(resolve => {
          finishScan = resolve
        })
    )

    const profileARequest = vi.fn(async () => ({ active_id: null, projects: [], repos: [], scoped_session_ids: [] }))
    const profileBRequest = vi.fn(async () => ({ active_id: null, projects: [], repos: [], scoped_session_ids: [] }))
    const profileAGateway = { connectionState: 'open', request: profileARequest }
    const profileBGateway = { connectionState: 'open', request: profileBRequest }

    desktopGit.mockReturnValue({ scanRepos } as never)
    activeGateway.mockReturnValue(profileAGateway as never)

    const pending = scanAndRecordRepos(true)

    await Promise.resolve()

    expect(scanRepos).toHaveBeenCalled()

    activeGateway.mockReturnValue(profileBGateway as never)
    finishScan([{ label: 'alpha', root: '/work/alpha' }])

    await pending

    expect(profileARequest).toHaveBeenCalledWith('projects.record_repos', {
      repos: [{ label: 'alpha', root: '/work/alpha' }]
    })
    expect(profileBRequest).not.toHaveBeenCalled()
  })

  it('does not publish a late tree response after the active gateway changes', async () => {
    let finishTree: (payload: {
      active_id: string
      projects: never[]
      scoped_session_ids: string[]
    }) => void = () => undefined

    const treeResponse = new Promise<{
      active_id: string
      projects: never[]
      scoped_session_ids: string[]
    }>(resolve => {
      finishTree = resolve
    })

    const profileARequest = vi.fn((method: string) => {
      if (method === 'projects.tree') {
        return treeResponse
      }

      return Promise.resolve({})
    })

    const profileAGateway = { connectionState: 'open', request: profileARequest }
    const profileBGateway = { connectionState: 'open', request: vi.fn() }

    desktopGit.mockReturnValue({ scanRepos: vi.fn(async () => []) } as never)
    activeGateway.mockReturnValue(profileAGateway as never)
    $projectTree.set([])
    $activeProjectId.set('profile-b-project')

    const pending = scanAndRecordRepos(true)

    await vi.waitFor(() => {
      expect(profileARequest).toHaveBeenCalledWith('projects.tree', { preview_limit: 3 })
    })
    activeGateway.mockReturnValue(profileBGateway as never)
    finishTree({ active_id: 'profile-a-project', projects: [], scoped_session_ids: [] })
    await pending

    expect($activeProjectId.get()).toBe('profile-b-project')
    expect($projectTree.get()).toEqual([])
  })
})
