import { cleanup, render } from '@testing-library/react'
import type { ReactNode } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { SessionInfo } from '@/hermes'
import type * as ProjectsStore from '@/store/projects'

import { SidebarSessionsSection } from '../sessions-section'

import type * as Model from './model'
import { readProjectRows, resolveProjectDropIntent } from './project-drag'
import type { SidebarProjectTree } from './workspace-groups'

vi.mock('../new-session-drag', () => ({ startNewProjectDrag: vi.fn(), startNewSessionDrag: vi.fn() }))

// Session rows are inert here — this file is about the project rows' geometry.
vi.mock('../session-row', () => ({
  SidebarSessionRow: ({ session }: { session: SessionInfo }) => <div data-session-row={session.id} />
}))

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      common: { cancel: 'Cancel' },
      desktop: {},
      profiles: { switchToProfile: (label: string) => `Switch to ${label}` },
      sidebar: {
        dateDivider: {},
        nav: { 'new-session': 'New session' },
        newSessionIn: (label: string) => `New session in ${label}`,
        noSessions: 'No sessions yet',
        projects: {
          autoDiscovered: 'Auto-discovered',
          copyPath: 'Copy path',
          dragNestInto: (label: string) => `Add subproject to ${label}`,
          dragTopLevel: 'Top level',
          enter: (label: string) => `Enter ${label}`,
          menu: 'Actions',
          menuNewSubproject: 'New subproject…',
          nestFailed: 'Could not move project',
          reorder: (label: string) => `Reorder ${label}`,
          reveal: 'Reveal in file manager',
          showAllCount: (count: number) => `Show all ${count} sessions`,
          toggle: (label: string, open: boolean) => `${open ? 'Show' : 'Hide'} ${label} sessions`
        },
        row: {},
        showMoreIn: (count: number, label: string) => `Show ${count} more in ${label}`
      },
      statusStack: { coding: { switchFailed: (label: string) => `Could not switch to ${label}` } }
    }
  })
}))

vi.mock('@/store/projects', async importOriginal => ({
  ...(await importOriginal<typeof ProjectsStore>()),
  fetchProjectSessions: vi.fn(),
  listRepoBranches: vi.fn().mockResolvedValue([]),
  projectProfile: () => 'default',
  removeWorktreePath: vi.fn(),
  switchBranchInRepo: vi.fn()
}))

vi.mock('./model', async () => ({
  ...(await vi.importActual<typeof Model>('./model')),
  latestProjectSessions: () => [],
  useWorkspaceNodeOpen: () => [true, vi.fn()]
}))

vi.mock('./project-menu', () => ({
  ProjectContextMenu: ({ children }: { children: ReactNode }) => children,
  ProjectMenu: () => null
}))

afterEach(cleanup)

const node = (id: string, over: Partial<SidebarProjectTree> = {}): SidebarProjectTree =>
  ({
    id,
    isNoProject: false,
    label: id,
    parentId: null,
    path: `/${id}`,
    previewSessions: [],
    repos: [],
    sessionCount: 1,
    ...over
  }) as SidebarProjectTree

const session = (id: string): SessionInfo => ({ id }) as SessionInfo

const renderOverview = (overview: SidebarProjectTree[], previews: Record<string, SessionInfo[]>) =>
  render(
    <SidebarSessionsSection
      activeSessionId={null}
      emptyState={null}
      label="Projects"
      onArchiveSession={vi.fn()}
      onDeleteSession={vi.fn()}
      onNewSessionInWorkspace={vi.fn()}
      onResumeSession={vi.fn()}
      onToggle={vi.fn()}
      onTogglePin={vi.fn()}
      onToggleUnread={vi.fn()}
      open
      pinned={false}
      projectOverview={overview}
      projectOverviewPreviews={previews}
      sessions={[]}
    />
  )

describe('project nest rows', () => {
  const parent = node('p_parent', { label: 'Parent' })
  const child = node('p_child', { label: 'Child', parentId: 'p_parent' })
  const other = node('p_other', { label: 'Other', parentId: '' })
  const projects = [parent, child, other]

  /** 18px a row with a 4px gap after each project block — the shape the real sidebar produces, where a
   *  project's block is taller than its own row and the space between two blocks is a reorder target.
   *  `draggedTop` slides one row up under the pointer, the way dnd-kit transforms the dragged item. */
  const lay = (elements: HTMLElement[], draggedTop: null | number = null) => {
    const tops = new Map<string, { bottom: number; top: number }>()
    let cursor = 0

    for (const el of elements) {
      const id = el.dataset.sessionsProject ?? ''
      const height = (1 + el.querySelectorAll('[data-session-row]').length) * 20
      const top = id === 'p_other' && draggedTop !== null ? draggedTop : cursor

      tops.set(id, { bottom: top + height, top })
      cursor += height + 4
    }

    vi.spyOn(HTMLElement.prototype, 'getBoundingClientRect').mockImplementation(function (this: HTMLElement) {
      const rect = tops.get(this.dataset.sessionsProject ?? '') ?? { bottom: 0, top: 0 }

      return {
        ...rect,
        height: rect.bottom - rect.top,
        left: 0,
        right: 220,
        toJSON: () => ({}),
        width: 220,
        x: 0,
        y: rect.top
      } as DOMRect
    })

    return tops
  }

  const renderNested = () =>
    renderOverview(projects, { p_child: [session('s_child')], p_parent: [session('s_parent')] })

  it('renders a nested project as the next row after its parent', () => {
    const { container } = renderNested()

    expect(
      [...container.querySelectorAll<HTMLElement>('[data-sessions-project]')].map(el => el.dataset.sessionsProject)
    ).toEqual(['p_parent', 'p_child', 'p_other'])
  })

  it('targets the row under the pointer, with the dragged row out of the geometry', () => {
    const { container } = renderNested()
    const elements = [...container.querySelectorAll<HTMLElement>('[data-sessions-project]')]

    // `p_other` is being dragged and dnd-kit has slid its row up under the pointer, inside the
    // parent's area. Its own area cannot be a target: the release would nest it into itself.
    lay(elements, 30)

    // It IS in the geometry the list renders — where dnd-kit put it — and is taken back out for the
    // drag it belongs to.
    expect(readProjectRows().find(row => row.id === 'p_other')?.box.top).toBe(30)

    const rows = readProjectRows('p_other')
    const parentRow = rows.find(row => row.id === 'p_parent')!
    const childRow = rows.find(row => row.id === 'p_child')!

    expect(rows.map(row => row.id)).toEqual(['p_parent', 'p_child'])
    // Every row is its own area: the parent's reaches its session row and stops there, at the gap
    // before the nested project — nothing is stretched over the space below it.
    expect(parentRow.box).toEqual({ bottom: 40, left: 0, right: 220, top: 0 })
    expect(childRow.box.top).toBeGreaterThan(parentRow.box.bottom)

    // A release on the child's own row names the child.
    expect(
      resolveProjectDropIntent({
        activeId: 'p_other',
        pointer: { x: 40, y: childRow.box.top + 10 },
        projects,
        rows
      })
    ).toEqual({ kind: 'into', targetId: 'p_child' })

    // A release on the parent's own session row is the parent: a project is as tall a target as it is
    // drawn.
    expect(
      resolveProjectDropIntent({
        activeId: 'p_other',
        pointer: { x: 40, y: parentRow.box.bottom - 10 },
        projects,
        rows
      })
    ).toEqual({ kind: 'into', targetId: 'p_parent' })

    // The gap between the two areas is still a reorder: the space under a project is not part of it,
    // so a release there must not fall back to "the area I am somewhere inside".
    expect(
      resolveProjectDropIntent({
        activeId: 'p_other',
        pointer: { x: 40, y: (parentRow.box.bottom + childRow.box.top) / 2 },
        projects,
        rows
      })
    ).toBeNull()
  })
})
