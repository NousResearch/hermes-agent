import { describe, expect, it } from 'vitest'

import type { SidebarProjectTree } from '@/app/chat/sidebar/projects/workspace-groups'
import type { SessionInfo } from '@/hermes'

import { cockpitProjects, liveSessionState, projectOverviewRoute, projectSessionList } from './model'

const session = (id: string, lastActive: number): SessionInfo =>
  ({ id, last_active: lastActive, started_at: lastActive }) as unknown as SessionInfo

const node = (id: string, extra: Partial<SidebarProjectTree> = {}): SidebarProjectTree => ({
  id,
  label: id,
  path: `/work/${id}`,
  repos: [],
  sessionCount: 0,
  ...extra
})

describe('projects cockpit model', () => {
  it('lists only real projects: never the Home bucket or archived ones', () => {
    const ids = cockpitProjects(
      [node('__no_project__', { isNoProject: true }), node('p_a'), node('p_old', { archived: true })],
      null
    ).map(project => project.id)

    expect(ids).toEqual(['p_a'])
  })

  it('keeps dismissed auto-discovered repositories out of the cockpit', () => {
    const ids = cockpitProjects([node('/work/auto', { isAuto: true }), node('p_a')], null, ['/work/auto']).map(
      project => project.id
    )

    expect(ids).toEqual(['p_a'])
  })

  it('prefers every backend-assigned session over the preview, newest first, without tombstoned rows', () => {
    const project = node('p_a', { previewSessions: [session('preview-only', 99)] })

    const hydrated = node('p_a', {
      repos: [
        {
          groups: [
            { id: 'main', label: 'main', path: '/work/p_a', sessions: [session('old', 1), session('gone', 5)] },
            { id: 'wt', label: 'feat', path: '/work/p_a-wt', sessions: [session('new', 9), session('old', 1)] }
          ],
          id: '/work/p_a',
          label: 'p_a',
          path: '/work/p_a',
          sessionCount: 3
        }
      ]
    })

    expect(projectSessionList(project, hydrated, new Set(['gone'])).map(s => s.id)).toEqual(['new', 'old'])
    expect(projectSessionList(project, null).map(s => s.id)).toEqual(['preview-only'])
  })

  it('reports a live state only for sessions with running work', () => {
    expect(liveSessionState(undefined)).toBeNull()
    expect(['idle', 'draft', 'unread'].map(state => liveSessionState(state as never))).toEqual([null, null, null])
    expect(liveSessionState('needs-input')).toBe('needs-input')
  })

  it('round-trips the selected project through the route', () => {
    const route = projectOverviewRoute('p_a b')

    expect(new URLSearchParams(route.split('?')[1]).get('project')).toBe('p_a b')
    expect(projectOverviewRoute(null)).toBe('/projects')
  })
})
