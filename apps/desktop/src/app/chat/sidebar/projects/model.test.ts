import { describe, expect, it } from 'vitest'

import { orderProjectsByIds, sortProjectsAlphabetically, sortProjectsForOverview } from './model'
import { NO_PROJECT_ID, type SidebarProjectTree } from './workspace-groups'

function makeProject(id: string, sessionCount: number): SidebarProjectTree {
  return {
    id,
    isAuto: true,
    label: id,
    lastActive: 0,
    path: `/repos/${id}`,
    previewSessions: [],
    repos: [],
    sessionCount
  }
}

const home = (): SidebarProjectTree => ({
  ...makeProject(NO_PROJECT_ID, 2),
  isAuto: false,
  isNoProject: true,
  path: null
})

const ids = (projects: SidebarProjectTree[]) => projects.map(project => project.id)

describe('orderProjectsByIds', () => {
  it('leaves the deterministic sort alone when nothing has been dragged', () => {
    const projects = [makeProject('a', 0), makeProject('b', 2)]

    expect(orderProjectsByIds(projects, [])).toBe(projects)
  })

  it('applies the saved manual order', () => {
    const projects = [makeProject('a', 1), makeProject('b', 1), makeProject('c', 1)]

    expect(ids(orderProjectsByIds(projects, ['c', 'a', 'b']))).toEqual(['c', 'a', 'b'])
  })

  it('keeps freshly-scanned zero-session repos below the hand-ordered list', () => {
    // The regression: a disk scan keeps finding git checkouts the user has
    // never opened in Hermes. Surfacing every unsaved id at the top buried the
    // projects they deliberately dragged into place.
    const projects = [makeProject('scanned-1', 0), makeProject('mine', 4), makeProject('scanned-2', 0)]

    expect(ids(orderProjectsByIds(projects, ['mine']))).toEqual(['mine', 'scanned-1', 'scanned-2'])
  })

  it('still surfaces a new project that has real activity', () => {
    // A project you just started working in should not sink beneath the saved
    // order — only the zero-session discoveries do.
    const projects = [makeProject('ordered', 1), makeProject('just-started', 3)]

    expect(ids(orderProjectsByIds(projects, ['ordered']))).toEqual(['just-started', 'ordered'])
  })

  it('drops ids that are no longer present', () => {
    const projects = [makeProject('a', 1)]

    expect(ids(orderProjectsByIds(projects, ['gone', 'a']))).toEqual(['a'])
  })

  it('keeps Home on top of a hand-picked order', () => {
    const projects = [makeProject('a', 1), home(), makeProject('b', 1)]

    expect(ids(orderProjectsByIds(projects, ['b', 'a']))).toEqual([NO_PROJECT_ID, 'b', 'a'])
  })
})

describe('sortProjectsForOverview', () => {
  it('puts Home above the active project', () => {
    const active = { ...makeProject('active', 5), isAuto: false }
    const projects = [makeProject('scanned', 0), active, home()]

    expect(ids(sortProjectsForOverview(projects, 'active'))).toEqual([NO_PROJECT_ID, 'active', 'scanned'])
  })
})

describe('sortProjectsAlphabetically', () => {
  it('puts Home first, then every project A→Z by name, ignoring activity and auto/explicit', () => {
    const zeta = { ...makeProject('zeta', 9), isAuto: false, label: 'Zeta', lastActive: 999 }
    const alpha = { ...makeProject('alpha', 0), label: 'alpha' }
    const v10 = { ...makeProject('v10', 3), label: 'Project 10' }
    const v9 = { ...makeProject('v9', 3), label: 'Project 9' }

    expect(ids(sortProjectsAlphabetically([zeta, v10, home(), alpha, v9]))).toEqual([
      NO_PROJECT_ID,
      'alpha',
      'v9',
      'v10',
      'zeta'
    ])
  })

  it('breaks equal labels by id and does not mutate its input', () => {
    const input = [{ ...makeProject('b', 1), label: 'Same' }, { ...makeProject('a', 1), label: 'Same' }]

    expect(ids(sortProjectsAlphabetically(input))).toEqual(['a', 'b'])
    expect(ids(input)).toEqual(['b', 'a'])
  })

  it('differs from the stock overview sort, which leads with the active project', () => {
    const projects = [{ ...makeProject('alpha', 1), label: 'alpha' }, { ...makeProject('zeta', 1), isAuto: false, label: 'zeta' }]

    expect(ids(sortProjectsForOverview(projects, 'zeta'))).toEqual(['zeta', 'alpha'])
    expect(ids(sortProjectsAlphabetically(projects))).toEqual(['alpha', 'zeta'])
  })
})
