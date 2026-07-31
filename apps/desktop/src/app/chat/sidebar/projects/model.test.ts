import { renderHook, waitFor } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import { desktopGit } from '@/lib/desktop-git'

import { orderProjectsByIds, sortProjectsForOverview, useRepoWorktreeMap } from './model'
import { NO_PROJECT_ID, type SidebarProjectTree } from './workspace-groups'

vi.mock('@/lib/desktop-git', () => ({ desktopGit: vi.fn() }))

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

const withProbe = (worktreeList: (repoPath: string) => Promise<unknown>) =>
  vi.mocked(desktopGit).mockReturnValue({ worktreeList } as unknown as ReturnType<typeof desktopGit>)

// The lane layer treats an EMPTY worktree list as git's own "this folder is not
// a repository" verdict. A probe that never got an answer must therefore stay
// distinguishable from it, or a transient git failure silently restructures a
// real repo's lanes.
describe('useRepoWorktreeMap', () => {
  it('records a failed probe as unknown, not as an empty repo', async () => {
    withProbe(() => Promise.reject(new Error('git invocation failed')))

    const { result } = renderHook(() => useRepoWorktreeMap(['/repo'], true))

    await waitFor(() => expect(result.current[1]).toBe(false))
    expect(result.current[0]['/repo']).toBeNull()
  })

  it("keeps git's empty answer as an empty list", async () => {
    withProbe(() => Promise.resolve([]))

    const { result } = renderHook(() => useRepoWorktreeMap(['/plain'], true))

    await waitFor(() => expect(result.current[1]).toBe(false))
    expect(result.current[0]['/plain']).toEqual([])
  })
})
