import { expect, it } from 'vitest'

import { matchesSessionTags } from '@/lib/session-tags'
import type { SessionInfo } from '@/types/hermes'

import { excludeProjectSessions, reconcileEnteredProjectSessions, type SidebarProjectTree } from './workspace-groups'

it('preserves tags and filters overview previews and hydrated drill-in lanes identically', () => {
  const tagged = { id: 'older', tags: ['work'] } as SessionInfo
  const untagged = { id: 'other' } as SessionInfo

  const project: SidebarProjectTree = {
    id: 'p',
    label: 'Project',
    path: '/repo',
    sessionCount: 2,
    previewSessions: [tagged, untagged],
    repos: [
      {
        id: 'r',
        label: 'Repo',
        path: '/repo',
        sessionCount: 2,
        groups: [{ id: 'g', label: 'main', path: '/repo', sessions: [tagged, untagged] }]
      }
    ]
  }

  const filtered = excludeProjectSessions(project, session => !matchesSessionTags(session, ['work']))
  expect(filtered.previewSessions).toEqual([tagged])
  expect(filtered.repos[0].groups[0].sessions).toEqual([tagged])
  expect(reconcileEnteredProjectSessions([], filtered.previewSessions)).toEqual([tagged])
  expect(project.repos[0].groups[0].sessions).toHaveLength(2)
})
