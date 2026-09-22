import { cleanup, render } from '@testing-library/react'
import type * as React from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { SessionInfo } from '@/hermes'

import { SidebarSessionsSection } from './sessions-section'
import type { SidebarProjectTree } from './projects/workspace-groups'

afterEach(cleanup)

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      sidebar: {
        dateDivider: {
          earlierThisMonth: 'Earlier this month',
          lastMonth: 'Last month',
          lastWeek: 'Last week',
          older: 'Older',
          today: 'Today',
          yesterday: 'Yesterday'
        },
        projects: {
          workspaceArchived: 'Archived',
          workspaceShared: 'Shared'
        }
      }
    }
  })
}))

vi.mock('./virtual-session-list', () => ({
  VirtualSessionList: () => null
}))

vi.mock('./session-row', () => ({
  SidebarSessionRow: ({ session }: { session: SessionInfo }) => (
    <div data-testid={`session-row-${session.id}`}>{session.id}</div>
  )
}))

vi.mock('./projects/overview-row', () => ({
  ProjectOverviewRow: ({ project }: { project: SidebarProjectTree }) => (
    <div data-testid={`project-row-${project.id}`}>{project.label}</div>
  )
}))

function makeProject(
  id: string,
  overrides: Partial<SidebarProjectTree> = {}
): SidebarProjectTree {
  return {
    id,
    label: id,
    path: `/home/morten/${id}`,
    repos: [],
    sessionCount: 0,
    ...overrides
  } as SidebarProjectTree
}

const noop = () => {}

describe('project overview grouped by owner (all-profiles)', () => {
  it('puts each project under its owner profile, archived under the owner Arkivet sub-group', () => {
    const overview = [
      makeProject('home', { label: 'Home', isNoProject: true }),
      makeProject('res-active', { profiles: ['researcher'] }),
      makeProject('res-old', { profiles: ['researcher'], archived: true }),
      makeProject('def-active', { profiles: ['default'] }),
      makeProject('shared-row', { profiles: ['energy-rent', 'byopus'] }),
      makeProject('res-old-2', { profiles: ['researcher'], archived: true })
    ]

    const { container } = render(
      <SidebarSessionsSection
        activeSessionId={null}
        emptyState={<div>Empty</div>}
        label="Sessions"
        onArchiveSession={noop}
        onDeleteSession={noop}
        onResumeSession={noop}
        onToggle={noop}
        onTogglePin={noop}
        onToggleUnread={noop}
        open={true}
        pinned={false}
        projectOverview={overview}
        projectOverviewProfiles={true}
        sessions={[]}
      />
    )

    console.log('HTML:', container.innerHTML)
    const text = container.textContent ?? ''
    const idx = (needle: string) => {
      const i = text.indexOf(needle)
      expect(i, `fant ikke "${needle}" i: ${text}`).toBeGreaterThanOrEqual(0)
      return i
    }

    // Home foerst, deretter gruppene med eier-overskrift.
    expect(idx('Home')).toBeLessThan(idx('researcher'))
    expect(idx('researcher')).toBeLessThan(idx('res-active'))

    // Arkivet er eierens undergruppe: etter researcher-radene, foer neste profil.
    expect(idx('res-active')).toBeLessThan(idx('Archived · 2'))
    expect(idx('Archived · 2')).toBeLessThan(idx('res-old'))
    expect(idx('res-old')).toBeLessThan(idx('default'))

    // Fler-eiers-rader havner i en delt gruppe med eierlisten.
    expect(idx('default')).toBeLessThan(idx('Shared · byopus, energy-rent'))
    expect(idx('Shared · byopus, energy-rent')).toBeLessThan(idx('shared-row'))
  })

  it('renders the flat overview as before when grouping is off', () => {
    const overview = [
      makeProject('home', { label: 'Home', isNoProject: true }),
      makeProject('a', { profiles: ['researcher'], archived: true })
    ]

    const { container } = render(
      <SidebarSessionsSection
        activeSessionId={null}
        emptyState={<div>Empty</div>}
        label="Sessions"
        onArchiveSession={noop}
        onDeleteSession={noop}
        onResumeSession={noop}
        onToggle={noop}
        onTogglePin={noop}
        onToggleUnread={noop}
        open={true}
        pinned={false}
        projectOverview={overview}
        projectOverviewProfiles={false}
        sessions={[]}
      />
    )

    const text = container.textContent ?? ''
    expect(text).toContain('Archived · 1')
    expect(text).not.toContain('researcher')
  })
})
