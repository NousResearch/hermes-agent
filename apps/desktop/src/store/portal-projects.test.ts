// @vitest-environment jsdom
import { beforeEach, describe, expect, it, vi } from 'vitest'

const requestGatewayForAgent = vi.fn()
const moveSessionToProject = vi.fn().mockResolvedValue(undefined)
const refreshProjects = vi.fn().mockResolvedValue(undefined)
const refreshProjectTree = vi.fn().mockResolvedValue(undefined)

vi.mock('@/store/gateway', () => ({
  activeGatewayConnectionId: () => null,
  requestGatewayForAgent: (...args: unknown[]) => requestGatewayForAgent(...args)
}))
vi.mock('@/store/profile', () => ({
  $activeGatewayProfile: { get: () => 'default' }
}))
vi.mock('@/store/projects', () => ({
  refreshProjects: () => refreshProjects(),
  refreshProjectTree: () => refreshProjectTree()
}))
vi.mock('@/store/notifications', () => ({
  notifyError: vi.fn()
}))
vi.mock('@/store/session', () => ({
  $sessions: { get: () => [], set: vi.fn() },
  sessionMatchesStoredId: (row: { id: string }, id: string) => row.id === id
}))

import { agentSidebarKey } from '@/store/agent-sidebar-preferences'
import {
  $pendingPortalProjectForNewChat,
  $portalAgentProjects,
  $portalProjectExpanded,
  agentPortalProjectsFor,
  armPortalProjectNewChat,
  claimPendingPortalProjectSession,
  createPortalProject,
  deletePortalProject,
  mergeBackendPortalProjects,
  moveSessionToPortalProject,
  PORTAL_PROJECT_NAME_MAX,
  rekeyPortalAgentProjects,
  renamePortalProject,
  resolvePortalProjectForSession
} from '@/store/portal-projects'
import {
  addPortalProjectFolderSynced,
  createPortalProjectSynced,
  setPortalProjectAppearanceSynced,
  syncPortalAgentProjects
} from '@/store/portal-projects-sync'
const owner = { connectionId: null, profile: 'default' }

beforeEach(() => {
  $portalAgentProjects.set({})
  $portalProjectExpanded.set({})
  requestGatewayForAgent.mockReset()
  moveSessionToProject.mockClear()
  refreshProjects.mockClear()
  refreshProjectTree.mockClear()
})

describe('portal-projects store', () => {
  it('creates an empty project (Flow B) and rejects empty or >25 char names', () => {
    expect(createPortalProject(owner, '   ')).toBeNull()
    expect(createPortalProject(owner, 'a'.repeat(PORTAL_PROJECT_NAME_MAX + 1))).toBeNull()

    const created = createPortalProject(owner, '  Q4 planning  ')
    expect(created).not.toBeNull()
    expect(created?.project.name).toBe('Q4 planning')

    const entry = agentPortalProjectsFor($portalAgentProjects.get(), owner)
    expect(entry.projects).toHaveLength(1)
    expect(entry.sessionProject).toEqual({})
  })

  it('creates a project and moves a chat into it (Flow A), and Undo deletes + restores membership', () => {
    const first = createPortalProject(owner, 'Existing', 's1')!
    const created = createPortalProject(owner, 'Design reviews', 's1')!

    expect(created.previousProjectId).toBe(first.project.id)
    expect(agentPortalProjectsFor($portalAgentProjects.get(), owner).sessionProject.s1).toBe(created.project.id)

    deletePortalProject(owner, created.project.id, {
      previousProjectId: created.previousProjectId,
      sessionId: 's1'
    })

    const afterUndo = agentPortalProjectsFor($portalAgentProjects.get(), owner)
    expect(afterUndo.projects.map(p => p.name)).toEqual(['Existing'])
    expect(afterUndo.sessionProject.s1).toBe(first.project.id)
  })

  it('moves chats into a project, removes them back to loose chats, renames, and rekeys on agent rename', () => {
    const p1 = createPortalProject(owner, 'Launch prep')!.project
    moveSessionToPortalProject(owner, 's2', p1.id)
    expect(agentPortalProjectsFor($portalAgentProjects.get(), owner).sessionProject.s2).toBe(p1.id)

    expect(renamePortalProject(owner, p1.id, 'Launch v2')).toBe(true)
    expect(agentPortalProjectsFor($portalAgentProjects.get(), owner).projects[0].name).toBe('Launch v2')

    moveSessionToPortalProject(owner, 's2', null)
    expect(agentPortalProjectsFor($portalAgentProjects.get(), owner).sessionProject.s2).toBeUndefined()

    const fromKey = agentSidebarKey(owner)
    const toOwner = { connectionId: null, profile: 'ops' }
    rekeyPortalAgentProjects(fromKey, agentSidebarKey(toOwner))
    expect(agentPortalProjectsFor($portalAgentProjects.get(), owner).projects).toHaveLength(0)
    expect(agentPortalProjectsFor($portalAgentProjects.get(), toOwner).projects[0].name).toBe('Launch v2')
  })

  it('arms from an existing chat without prematurely disarming and files the newly created session', () => {
    const p1 = createPortalProject(owner, 'Q4 planning')!.project
    armPortalProjectNewChat(owner, p1.id, ['existing-1'], 'existing-1')
    expect($portalProjectExpanded.get()[p1.id]).toBe(true)
    expect($pendingPortalProjectForNewChat.get()?.armedAtSessionId).toBe('existing-1')
    expect($pendingPortalProjectForNewChat.get()?.leftArmedSession).toBe(false)

    expect(claimPendingPortalProjectSession(owner, 'existing-1')).toBe(false)
    expect(claimPendingPortalProjectSession({ connectionId: null, profile: 'other' }, 'new-1')).toBe(false)

    expect(claimPendingPortalProjectSession(owner, 'new-1')).toBe(true)
    expect($pendingPortalProjectForNewChat.get()).toBeNull()
    expect(agentPortalProjectsFor($portalAgentProjects.get(), owner).sessionProject['new-1']).toBe(p1.id)
  })

  it('merges backend projects.db records and matches sessions by longest workspace folder prefix', () => {
    mergeBackendPortalProjects(owner, [
      {
        color: '#22c55e',
        created_at: 100,
        folders: [{ is_primary: true, path: '/work' }],
        icon: 'repo',
        id: 'p_root',
        name: 'Work',
        primary_path: '/work',
        slug: 'work'
      },
      {
        created_at: 200,
        folders: [{ is_primary: true, path: '/work/hermes' }],
        id: 'p_hermes',
        name: 'Hermes',
        primary_path: '/work/hermes',
        slug: 'hermes'
      }
    ])

    const entry = agentPortalProjectsFor($portalAgentProjects.get(), owner)
    expect(entry.projects.map(p => [p.id, p.path, p.color, p.icon, p.backend])).toEqual([
      ['p_root', '/work', '#22c55e', 'repo', true],
      ['p_hermes', '/work/hermes', null, null, true]
    ])

    expect(resolvePortalProjectForSession(entry, { cwd: '/work/hermes/apps/desktop', id: 's-nested' })).toBe('p_hermes')
    expect(resolvePortalProjectForSession(entry, { cwd: '/work/other', id: 's-parent' })).toBe('p_root')
    expect(resolvePortalProjectForSession(entry, { cwd: '/tmp', id: 's-outside' })).toBeNull()

    // Explicit override takes precedence over cwd matching.
    moveSessionToPortalProject(owner, 's-nested', 'p_root')
    const updated = agentPortalProjectsFor($portalAgentProjects.get(), owner)
    expect(resolvePortalProjectForSession(updated, { cwd: '/work/hermes/apps/desktop', id: 's-nested' })).toBe('p_root')

    // Explicit detach prevents cwd prefix matching from re-filing the chat back into its folder project.
    moveSessionToPortalProject(owner, 's-nested', null, { detach: true })
    const detached = agentPortalProjectsFor($portalAgentProjects.get(), owner)
    expect(resolvePortalProjectForSession(detached, { cwd: '/work/hermes/apps/desktop', id: 's-nested' })).toBeNull()

    // Identical backend merge is a no-op (preserves store reference identity).
    const beforeRef = $portalAgentProjects.get()
    mergeBackendPortalProjects(owner, [
      {
        color: '#22c55e',
        created_at: 999,
        folders: [{ is_primary: true, path: '/work' }],
        icon: 'repo',
        id: 'p_root',
        name: 'Work',
        primary_path: '/work',
        slug: 'work'
      },
      {
        created_at: 999,
        folders: [{ is_primary: true, path: '/work/hermes' }],
        id: 'p_hermes',
        name: 'Hermes',
        primary_path: '/work/hermes',
        slug: 'hermes'
      }
    ])
    expect($portalAgentProjects.get()).toBe(beforeRef)
  })

  it('syncs folder-backed & folder-less creation, appearance, and folder attachment through gateway RPCs', async () => {
    requestGatewayForAgent
      .mockResolvedValueOnce({
        project: {
          created_at: 300,
          folders: [{ is_primary: true, path: '/repo/app' }],
          id: 'p_app',
          name: 'App',
          primary_path: '/repo/app',
          slug: 'app'
        }
      })
      .mockResolvedValueOnce({})

    const created = await createPortalProjectSynced(owner, 'App', {
      folder: '/repo/app',
      moveSessionId: 's-move'
    })
    expect(created?.project.id).toBe('p_app')
    expect(created?.project.path).toBe('/repo/app')
    expect(created?.project.backend).toBe(true)
    expect(requestGatewayForAgent).toHaveBeenNthCalledWith(
      1,
      null,
      'default',
      'projects.create',
      { folders: ['/repo/app'], name: 'App', primary_path: '/repo/app', profile: 'default' },
      8000,
      undefined,
      { spawnPriority: 'foreground' }
    )
    expect(requestGatewayForAgent).toHaveBeenNthCalledWith(
      2,
      null,
      'default',
      'session.workspace.move',
      { cwd: '/repo/app', profile: 'default', session_key: 's-move' },
      8000,
      undefined,
      { spawnPriority: 'foreground' }
    )
    requestGatewayForAgent.mockResolvedValueOnce({})
    setPortalProjectAppearanceSynced(owner, created!.project, { color: '#f59e0b', icon: 'rocket' })
    expect(agentPortalProjectsFor($portalAgentProjects.get(), owner).projects[0]).toMatchObject({
      color: '#f59e0b',
      icon: 'rocket'
    })

    requestGatewayForAgent.mockResolvedValueOnce({})
    addPortalProjectFolderSynced(owner, created!.project, '/repo/docs')
    expect(agentPortalProjectsFor($portalAgentProjects.get(), owner).projects[0].folders).toEqual([
      '/repo/app',
      '/repo/docs'
    ])

    requestGatewayForAgent.mockResolvedValueOnce({
      projects: [
        {
          created_at: 400,
          folders: [],
          id: 'p_notes',
          name: 'Q4 notes',
          primary_path: null,
          slug: 'q4-notes'
        }
      ]
    })
    await syncPortalAgentProjects(owner)
    expect(agentPortalProjectsFor($portalAgentProjects.get(), owner).projects.map(p => p.id)).toEqual(['p_notes'])
  })
})
