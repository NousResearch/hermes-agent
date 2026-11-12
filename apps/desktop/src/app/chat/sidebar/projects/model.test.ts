import { act, renderHook } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import { orderProjectsByIds, sortProjectsForOverview, useRevealedRows } from './model'
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

describe('useRevealedRows', () => {
  // The reveal page is disposable: closing the group resets it, so reopening
  // starts back at the short first page instead of restoring a drilled-down
  // list (the sidebar's "collapse, then reopen shows the brief few" behavior).
  it('starts back at the first page after the group closes', () => {
    const rows = Array.from({ length: 60 }, (_, index) => index)

    const { result, rerender } = renderHook(({ open }) => useRevealedRows(rows, 5, open), {
      initialProps: { open: true }
    })

    act(() => result.current.showMore())
    expect(result.current.shown).toHaveLength(55)

    // Collapse, then reopen: the drill-down is gone.
    rerender({ open: false })
    rerender({ open: true })
    expect(result.current.shown).toHaveLength(5)
    expect(result.current.more).toBe(50)
  })

  it('keeps revealing within a group that never closed', () => {
    const rows = Array.from({ length: 60 }, (_, index) => index)
    const { result } = renderHook(() => useRevealedRows(rows, 5))

    act(() => result.current.showMore())
    expect(result.current.shown).toHaveLength(55)
    expect(result.current.more).toBe(5)
  })
})
