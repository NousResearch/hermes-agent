import { describe, expect, it } from 'vitest'

import {
  nestProjectsByParent,
  orderProjectsByIds,
  projectBackTarget,
  projectDepths,
  projectDescendantIds,
  projectSubtreeSessionIds,
  sortProjectsForOverview,
  visibleProjectRows
} from './model'
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

describe('projectDepths', () => {
  const nested = (id: string, parentId?: null | string) => ({ ...makeProject(id, 0), parentId })

  it('counts the parent chain: top level 0, child 1, grandchild 2', () => {
    const depths = projectDepths([nested('top'), nested('mid', 'top'), nested('leaf', 'mid')])

    expect(depths.get('top')).toBe(0)
    expect(depths.get('mid')).toBe(1)
    expect(depths.get('leaf')).toBe(2)
  })

  it('treats a parent that is not in the list as top level, like the rest of the model', () => {
    expect(projectDepths([nested('orphan', 'p_gone')]).get('orphan')).toBe(0)
    // An explicit top level ('') is not a parent either.
    expect(projectDepths([nested('explicit', '')]).get('explicit')).toBe(0)
  })

  it('terminates on a parent cycle instead of hanging', () => {
    const depths = projectDepths([nested('a', 'b'), nested('b', 'a')])

    expect(depths.get('a')).toBe(1)
    expect(depths.get('b')).toBe(1)
  })
})

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

describe('nestProjectsByParent', () => {
  // An explicit project keeps its own sessions; `parentId` only decides where
  // its row renders. The backend derives it from nested project folders.
  const child = (id: string, parentId: string): SidebarProjectTree => ({
    ...makeProject(id, 1),
    isAuto: false,
    parentId
  })

  it('moves a nested project directly under its parent', () => {
    const projects = [makeProject('other', 1), makeProject('dev', 1), child('align', 'dev')]

    expect(ids(nestProjectsByParent(projects))).toEqual(['other', 'dev', 'align'])
  })

  it('keeps Home first and the incoming order within a level', () => {
    const projects = [
      home(),
      makeProject('dev', 1),
      makeProject('other', 1),
      child('align', 'dev'),
      child('router', 'dev')
    ]

    expect(ids(nestProjectsByParent(projects))).toEqual([NO_PROJECT_ID, 'dev', 'align', 'router', 'other'])
  })

  it('nests a chain parent-first', () => {
    const projects = [child('align', 'dev'), child('dev', 'ws'), makeProject('ws', 1)]

    expect(ids(nestProjectsByParent(projects))).toEqual(['ws', 'dev', 'align'])
  })

  it('leaves a child at top level when its parent is gone', () => {
    // Hiding or dismissing the parent must not take the child's row with it.
    const projects = [makeProject('other', 1), child('align', 'gone')]

    expect(ids(nestProjectsByParent(projects))).toEqual(['other', 'align'])
  })

  it('ignores a self-referential parent id', () => {
    const projects = [child('align', 'align')]

    expect(nestProjectsByParent(projects)).toBe(projects)
  })

  it('returns the same list when nothing nests', () => {
    const projects = [makeProject('a', 1), makeProject('b', 1)]

    expect(nestProjectsByParent(projects)).toBe(projects)
  })

  it('never drops a row when parent ids form a cycle', () => {
    const projects = [child('a', 'b'), child('b', 'a')]

    expect(ids(nestProjectsByParent(projects)).sort()).toEqual(['a', 'b'])
  })
})

describe('projectDescendantIds', () => {
  it('walks the nest transitively and includes the project itself', () => {
    const projects = [
      makeProject('dev', 0),
      { ...makeProject('align', 0), parentId: 'dev' },
      { ...makeProject('leaf', 0), parentId: 'align' },
      makeProject('other', 0)
    ]

    expect([...projectDescendantIds(projects, 'dev')].sort()).toEqual(['align', 'dev', 'leaf'])
    expect([...projectDescendantIds(projects, 'leaf')]).toEqual(['leaf'])
    expect([...projectDescendantIds(projects, 'other')]).toEqual(['other'])
  })
})

describe('projectSubtreeSessionIds', () => {
  // Ownership is deepest-wins, so a subproject's sessions are absent from its
  // ancestor's own `sessionIds` — which is exactly why a collapsed parent has to
  // collect them to show a status.
  const projects = [
    { ...makeProject('dev', 1), sessionIds: ['s_dev'] },
    { ...makeProject('align', 1), parentId: 'dev', sessionIds: ['s_align'] },
    { ...makeProject('leaf', 0), parentId: 'align', sessionIds: ['s_leaf'] },
    { ...makeProject('other', 1), sessionIds: ['s_other'] }
  ]

  it('folds nested sessions into their ancestors', () => {
    expect(projectSubtreeSessionIds(projects, 'dev').sort()).toEqual(['s_align', 's_dev', 's_leaf'])
    expect(projectSubtreeSessionIds(projects, 'align').sort()).toEqual(['s_align', 's_leaf'])
    expect(projectSubtreeSessionIds(projects, 'other')).toEqual(['s_other'])
  })
})

describe('visibleProjectRows', () => {
  const child = (id: string, parentId: string): SidebarProjectTree => ({ ...makeProject(id, 0), parentId })

  const projects = [makeProject('dev', 0), child('align', 'dev'), child('leaf', 'align'), makeProject('other', 0)]

  const open =
    (...ids: string[]) =>
    (id: string) =>
      ids.includes(id)

  it('hides everything under a collapsed project, however deep', () => {
    expect(visibleProjectRows(projects, open('dev', 'align', 'other')).map(p => p.id)).toEqual([
      'dev',
      'align',
      'leaf',
      'other'
    ])
    // `align` is open, but its parent `dev` is closed — align and leaf go with it. `dev` itself
    // keeps its row: a project is only ever hidden by an ancestor, never by its own flag.
    expect(visibleProjectRows(projects, open('align', 'other')).map(p => p.id)).toEqual(['dev', 'other'])
    // The parent open again: only what its own closed child hides is gone.
    expect(visibleProjectRows(projects, open('dev', 'other')).map(p => p.id)).toEqual(['dev', 'align', 'other'])
  })

  it('leaves a project whose parent is not in the list standing on its own', () => {
    expect(visibleProjectRows([child('orphan', 'gone')], open('dev')).map(p => p.id)).toEqual(['orphan'])
  })
})

describe('projectBackTarget', () => {
  const child = (id: string, parentId: string): SidebarProjectTree => ({ ...makeProject(id, 0), parentId })

  it('steps back into the parent, so a drill-down walks out one level at a time', () => {
    const projects = [makeProject('dev', 0), child('align', 'dev'), child('leaf', 'align')]

    expect(projectBackTarget(projects, 'leaf')).toBe('align')
    expect(projectBackTarget(projects, 'align')).toBe('dev')
  })

  it('sends a top-level project straight to the overview', () => {
    const projects = [makeProject('dev', 0), child('align', 'dev')]

    expect(projectBackTarget(projects, 'dev')).toBeNull()
    // An explicit top level (`parentId: ''`) is top level too — nothing to step back into.
    expect(projectBackTarget([{ ...makeProject('top', 0), parentId: '' }], 'top')).toBeNull()
  })

  it('falls to the overview when the parent is no longer in the tree', () => {
    // A filtered-out or deleted parent is not a level the sidebar can render.
    expect(projectBackTarget([child('orphan', 'gone')], 'orphan')).toBeNull()
  })
})
