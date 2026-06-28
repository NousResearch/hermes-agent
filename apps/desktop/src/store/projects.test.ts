import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { HermesGateway } from '@/hermes'

const { fakeGateway, request } = vi.hoisted(() => {
  const request = vi.fn(async () => ({ projects: [], active_id: null, scoped_session_ids: [] }))
  const fakeGateway = {
    connectionState: 'open',
    request
  } as unknown as HermesGateway

  return {
    fakeGateway,
    request
  }
})

vi.mock('@/store/gateway', () => ({
  activeGateway: () => fakeGateway,
  ensureActiveGatewayOpen: vi.fn(async () => fakeGateway)
}))

import {
  $activeProjectId,
  $projectScope,
  $projects,
  $projectTree,
  $worktreeRefreshToken,
  ALL_PROJECTS,
  enterProject,
  exitProjectScope,
  fetchProjectSessions,
  refreshProjects,
  refreshProjectTree,
  refreshWorktrees
} from './projects'
import { $activeGatewayProfile } from './profile'

describe('project scope', () => {
  beforeEach(() => {
    window.localStorage.clear()
    request.mockClear()
    $projectScope.set(ALL_PROJECTS)
    $projects.set([])
    $projectTree.set([])
    $activeProjectId.set(null)
    $activeGatewayProfile.set('default')
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

  it('forwards the active profile to projects RPCs', async () => {
    $activeGatewayProfile.set('coder')

    await refreshProjects()
    await refreshProjectTree()
    await fetchProjectSessions('p_123')

    expect(request).toHaveBeenNthCalledWith(1, 'projects.list', { profile: 'coder' })
    expect(request).toHaveBeenNthCalledWith(2, 'projects.tree', { preview_limit: 3, profile: 'coder' })
    expect(request).toHaveBeenNthCalledWith(3, 'projects.project_sessions', {
      project_id: 'p_123',
      profile: 'coder'
    })
  })
})

describe('worktree refresh', () => {
  it('refreshWorktrees bumps the probe token so useRepoWorktreeMap refetches', () => {
    const before = $worktreeRefreshToken.get()
    refreshWorktrees()
    expect($worktreeRefreshToken.get()).toBe(before + 1)
  })
})
