import { cleanup, render, screen } from '@testing-library/react'
import type { ReactNode } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type * as ProjectsStore from '@/store/projects'
import { makeSessionInfo } from '@/test/session-info'
import type { SessionInfo } from '@/types/hermes'

import type * as Model from './model'
import { ProjectOverviewRow } from './overview-row'
import type { SidebarProjectTree } from './workspace-groups'

// The overview's own filtered preview (`projectOverviewPreviews`) is the only
// list that knows about optimistic removals. Its empty case used to fall back to
// the project tree's RAW lanes — the snapshot that still lists a session the user
// just archived/deleted — so the row came straight back under its project and
// stayed there until a reload rebuilt the tree. The fallback must honour the
// same exclusion the overlay applied.

afterEach(cleanup)

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      sidebar: {
        newSessionIn: (label: string) => `New session in ${label}`,
        projects: {
          enter: (label: string) => `Enter ${label}`,
          reorder: (label: string) => `Reorder ${label}`,
          toggle: (label: string, open: boolean) => `${open ? 'Show' : 'Hide'} ${label} sessions`,
          showAllCount: (count: number) => `Show all ${count} sessions`,
          autoDiscovered: 'Auto-discovered'
        },
        showMoreIn: (count: number, label: string) => `Show ${count} more in ${label}`
      }
    }
  })
}))

vi.mock('@/store/projects', async importOriginal => ({
  ...(await importOriginal<typeof ProjectsStore>()),
  fetchProjectSessions: vi.fn(async () => null),
  projectProfile: () => 'default'
}))

// Keep the real preview helpers (latestProjectSessions is the predicate under
// test); stub only the persisted open/collapse state.
vi.mock('./model', async () => ({
  ...(await vi.importActual<typeof Model>('./model')),
  useWorkspaceNodeOpen: () => [true, vi.fn()]
}))

vi.mock('./project-menu', () => ({
  ProjectContextMenu: ({ children }: { children: ReactNode }) => children,
  ProjectMenu: () => null
}))

const projectWith = (sessions: SessionInfo[]): SidebarProjectTree =>
  ({
    id: '/www/app',
    label: 'app',
    path: '/www/app',
    previewSessions: sessions,
    sessionCount: sessions.length,
    repos: [
      {
        id: '/www/app',
        label: 'app',
        path: '/www/app',
        sessionCount: sessions.length,
        groups: [{ id: '/www/app::branch::main', isMain: true, label: 'main', path: '/www/app', sessions }]
      }
    ]
  }) as unknown as SidebarProjectTree

const renderRows = (items: SessionInfo[]) => <div data-testid="rows">{items.map(item => item.id).join(',')}</div>

describe('ProjectOverviewRow removal fallback', () => {
  it('renders nothing when every preview row was just archived/deleted', () => {
    const archived = makeSessionInfo({ id: 'archived-row', message_count: 12 })
    // The overlay emptied the preview; the snapshot still lists the row.
    render(
      <ProjectOverviewRow
        isSessionHidden={session => session.id === 'archived-row'}
        previewSessions={[]}
        project={projectWith([archived])}
        renderRows={renderRows}
      />
    )

    expect(screen.queryByTestId('rows')).toBeNull()
  })

  it('drops only the removed rows from a snapshot fallback, keeping the rest', () => {
    const kept = makeSessionInfo({ id: 'kept-row', message_count: 4 })
    const archived = makeSessionInfo({ id: 'archived-row', message_count: 12 })

    render(
      <ProjectOverviewRow
        isSessionHidden={session => session.id === 'archived-row'}
        project={projectWith([archived, kept])}
        renderRows={renderRows}
      />
    )

    expect(screen.getByTestId('rows').textContent).toBe('kept-row')
  })
})
