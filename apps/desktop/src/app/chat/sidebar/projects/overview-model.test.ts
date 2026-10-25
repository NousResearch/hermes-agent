import { describe, expect, it } from 'vitest'

import { buildProjectOverview } from './overview-model'
import { NO_PROJECT_ID, type SidebarProjectTree } from './workspace-groups'

const project = (id: string, label: string, lastActive: number, extra: Partial<SidebarProjectTree> = {}) =>
  ({ id, label, lastActive, path: `/repos/${id}`, repos: [], sessionCount: 1, ...extra }) as SidebarProjectTree

const tree = () => [
  project('zulu', 'zulu-app', 30, { isAuto: false }),
  project('alpha', 'alpha-app', 10, { isAuto: false }),
  project(NO_PROJECT_ID, 'Home', 0, { isAuto: false, isNoProject: true, path: null }),
  project('mango', 'mango-app', 20, { isAuto: false })
]

const build = (alphabetical: boolean, projectOrderIds: string[] = []) =>
  buildProjectOverview({
    activeProjectId: null,
    alphabetical,
    homeLabel: 'Home',
    isHidden: () => false,
    orderRepos: repos => repos,
    projectFilter: [],
    projectOrderIds,
    projects: tree()
  }).map(p => p.id)

describe('buildProjectOverview', () => {
  it('keeps the stock order by default: Home, then recent activity, then the saved drag-order', () => {
    expect(build(false)).toEqual([NO_PROJECT_ID, 'zulu', 'mango', 'alpha'])
    expect(build(false, ['alpha', 'mango', 'zulu'])).toEqual([NO_PROJECT_ID, 'alpha', 'mango', 'zulu'])
  })

  it('lists Home first then A→Z when alphabetical, ignoring the saved drag-order', () => {
    expect(build(true)).toEqual([NO_PROJECT_ID, 'alpha', 'mango', 'zulu'])
    expect(build(true, ['zulu', 'mango', 'alpha'])).toEqual([NO_PROJECT_ID, 'alpha', 'mango', 'zulu'])
  })
})
