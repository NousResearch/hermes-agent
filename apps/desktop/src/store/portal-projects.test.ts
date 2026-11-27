// @vitest-environment jsdom
import { beforeEach, describe, expect, it } from 'vitest'

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
  moveSessionToPortalProject,
  PORTAL_PROJECT_NAME_MAX,
  rekeyPortalAgentProjects,
  renamePortalProject
} from '@/store/portal-projects'

const owner = { connectionId: null, profile: 'default' }

beforeEach(() => {
  $portalAgentProjects.set({})
  $portalProjectExpanded.set({})
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

  it('arms and automatically files a newly created session into the chosen project without stealing baseline chats', () => {
    const p1 = createPortalProject(owner, 'Q4 planning')!.project
    armPortalProjectNewChat(owner, p1.id, ['existing-1'])
    expect($portalProjectExpanded.get()[p1.id]).toBe(true)

    expect(claimPendingPortalProjectSession(owner, 'existing-1')).toBe(false)
    expect(claimPendingPortalProjectSession({ connectionId: null, profile: 'other' }, 'new-1')).toBe(false)

    expect(claimPendingPortalProjectSession(owner, 'new-1')).toBe(true)
    expect($pendingPortalProjectForNewChat.get()).toBeNull()
    expect(agentPortalProjectsFor($portalAgentProjects.get(), owner).sessionProject['new-1']).toBe(p1.id)
  })
})
