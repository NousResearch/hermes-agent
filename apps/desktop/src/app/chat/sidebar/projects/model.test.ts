import { describe, expect, it } from 'vitest'

import { enteredProjectHasContent, orderProjectsByIds, sortProjectsForOverview } from './model'
import {
  NO_PROJECT_ID,
  type SidebarProjectTree,
  type SidebarSessionGroup,
  type SidebarWorkspaceTree
} from './workspace-groups'

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

describe('enteredProjectHasContent', () => {
  const lane = (sessions: number): SidebarSessionGroup => ({
    id: 'lane',
    label: 'main',
    path: '/repos/p',
    sessions: Array.from({ length: sessions }, (_, index) => ({ id: `s${index}` }) as never)
  })

  const repo = (lanes: SidebarSessionGroup[]): SidebarWorkspaceTree => ({
    id: 'repo',
    label: 'repo',
    path: '/repos/p',
    groups: lanes,
    sessionCount: lanes.reduce((total, group) => total + group.sessions.length, 0)
  })

  // The overview tree ships its lanes WITHOUT rows (`hydrate=False`): rendering
  // that fallback as content is what left an entered project showing branch
  // headers with nothing under them.
  it('refuses the structure-only fallback until rows arrive', () => {
    const structureOnly = { ...makeProject('p', 43), repos: [repo([lane(0)])] }

    expect(enteredProjectHasContent(structureOnly, false)).toBe(false)
    // …and the very same node, once it carries hydrated rows, is content again.
    expect(enteredProjectHasContent({ ...structureOnly, repos: [repo([lane(3)])] }, false)).toBe(true)
    // A hydrated project whose rows were all filtered out (pinned) is still
    // content: the lanes are real, so the drill-in must keep rendering them.
    expect(enteredProjectHasContent({ ...structureOnly, repos: [repo([lane(0)])] }, true)).toBe(true)
    expect(enteredProjectHasContent({ ...makeProject('p', 5), repos: [] }, true)).toBe(true)
  })
})
