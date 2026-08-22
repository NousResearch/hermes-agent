import { cleanup, render } from '@testing-library/react'
import type * as React from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { SessionInfo } from '@/hermes'

import { SidebarSessionsSection, VIRTUALIZE_THRESHOLD } from './sessions-section'
import type { VirtualSessionListProps } from './virtual-session-list'

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
        }
      }
    }
  })
}))

const mockVirtualListPropsHistory: VirtualSessionListProps[] = []

vi.mock('./virtual-session-list', () => ({
  VirtualSessionList: (props: VirtualSessionListProps) => {
    mockVirtualListPropsHistory.push(props)

    return <div data-testid="virtual-session-list">Virtual List ({props.rows.length} rows)</div>
  }
}))

vi.mock('./session-row', () => ({
  SidebarSessionRow: ({
    branchStem,
    reorderable,
    session
  }: {
    branchStem?: string
    reorderable?: boolean
    session: SessionInfo
  }) => (
    <div
      data-branch-stem={branchStem ?? ''}
      data-reorderable={String(Boolean(reorderable))}
      data-testid={`session-row-${session.id}`}
    >
      {branchStem ?? ''}
      {session.id}
    </div>
  )
}))

vi.mock('./projects', () => ({
  EnteredProjectContent: () => null,
  ProjectOverviewRow: () => null,
  SidebarWorkspaceGroup: ({
    group,
    renderRows
  }: {
    group: { id: string; sessions: SessionInfo[] }
    renderRows: (sessions: SessionInfo[]) => React.ReactNode
  }) => <div data-testid={`profile-group-${group.id}`}>{renderRows(group.sessions)}</div>
}))

function makeSession(id: string, startedAt = 1000): SessionInfo {
  return {
    handoff_platform: null,
    handoff_state: null,
    id,
    last_active: startedAt,
    profile: 'default',
    started_at: startedAt
  } as unknown as SessionInfo
}

function generateSessions(count: number): SessionInfo[] {
  return Array.from({ length: count }, (_, i) => makeSession(`session-${i + 1}`, 10000 - i * 100))
}

const noop = () => {}

describe('SidebarSessionsSection memoization & virtualizer stability', () => {
  it('memoizes flatRows and passes the exact same rows array reference across parent re-renders', () => {
    mockVirtualListPropsHistory.length = 0

    const sessions = generateSessions(VIRTUALIZE_THRESHOLD + 5)

    const { rerender } = render(
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
        sessions={sessions}
      />
    )

    expect(mockVirtualListPropsHistory.length).toBe(1)
    const initialRowsRef = mockVirtualListPropsHistory[0].rows
    expect(initialRowsRef.length).toBeGreaterThan(VIRTUALIZE_THRESHOLD)

    // Re-render parent with the exact same sessions array and props
    rerender(
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
        sessions={sessions}
      />
    )

    expect(mockVirtualListPropsHistory.length).toBe(2)
    const nextRowsRef = mockVirtualListPropsHistory[1].rows

    // Confirm that the flatRows array reference remains strictly identical across renders (useMemo proof)
    expect(nextRowsRef).toBe(initialRowsRef)
  })

  it('re-computes flatRows reference when grouping or sessions change', () => {
    mockVirtualListPropsHistory.length = 0

    const initialSessions = generateSessions(VIRTUALIZE_THRESHOLD + 2)

    const { rerender } = render(
      <SidebarSessionsSection
        activeSessionId={null}
        emptyState={<div>Empty</div>}
        grouping="none"
        label="Sessions"
        onArchiveSession={noop}
        onDeleteSession={noop}
        onResumeSession={noop}
        onToggle={noop}
        onTogglePin={noop}
        onToggleUnread={noop}
        open={true}
        pinned={false}
        sessions={initialSessions}
      />
    )

    const firstRowsRef = mockVirtualListPropsHistory[0].rows

    // Switch on date dividers
    rerender(
      <SidebarSessionsSection
        activeSessionId={null}
        emptyState={<div>Empty</div>}
        grouping="date"
        label="Sessions"
        onArchiveSession={noop}
        onDeleteSession={noop}
        onResumeSession={noop}
        onToggle={noop}
        onTogglePin={noop}
        onToggleUnread={noop}
        open={true}
        pinned={false}
        sessions={initialSessions}
      />
    )

    const secondRowsRef = mockVirtualListPropsHistory[1].rows
    expect(secondRowsRef).not.toBe(firstRowsRef)

    // Change sessions array identity
    const updatedSessions = generateSessions(VIRTUALIZE_THRESHOLD + 4)
    rerender(
      <SidebarSessionsSection
        activeSessionId={null}
        emptyState={<div>Empty</div>}
        grouping="date"
        label="Sessions"
        onArchiveSession={noop}
        onDeleteSession={noop}
        onResumeSession={noop}
        onToggle={noop}
        onTogglePin={noop}
        onToggleUnread={noop}
        open={true}
        pinned={false}
        sessions={updatedSessions}
      />
    )

    const thirdRowsRef = mockVirtualListPropsHistory[2].rows
    expect(thirdRowsRef).not.toBe(secondRowsRef)
  })
})

describe('SidebarSessionsSection all-profile manual ordering', () => {
  it('makes rows reorderable inside independent profile groups in Manual mode', () => {
    const onReorderProfileSessions = vi.fn()
    const alpha = [makeSession('alpha-1'), makeSession('alpha-2')].map(session => ({ ...session, profile: 'alpha' }))
    const beta = [makeSession('beta-1'), makeSession('beta-2')].map(session => ({ ...session, profile: 'beta' }))

    render(
      <SidebarSessionsSection
        activeSessionId={null}
        emptyState={<div>Empty</div>}
        groups={[
          { id: 'alpha', label: 'Alpha', mode: 'profile', path: null, sessions: alpha },
          { id: 'beta', label: 'Beta', mode: 'profile', path: null, sessions: beta }
        ]}
        label="Sessions"
        manualOrderIdsByProfile={{
          alpha: ['alpha-2', 'alpha-1'],
          beta: ['beta-1', 'beta-2']
        }}
        onArchiveSession={noop}
        onDeleteSession={noop}
        onReorderProfileSessions={onReorderProfileSessions}
        onResumeSession={noop}
        onToggle={noop}
        onTogglePin={noop}
        onToggleUnread={noop}
        open
        pinned={false}
        sessions={[]}
      />
    )

    expect(globalThis.document.querySelectorAll('[data-reorderable="true"]')).toHaveLength(4)
    expect(
      [
        ...(globalThis.document
          .querySelector('[data-testid="profile-group-alpha"]')
          ?.querySelectorAll('[data-testid^="session-row-"]') ?? [])
      ].map(row => row.textContent)
    ).toEqual(['alpha-2', 'alpha-1'])
    expect(
      [
        ...(globalThis.document
          .querySelector('[data-testid="profile-group-beta"]')
          ?.querySelectorAll('[data-testid^="session-row-"]') ?? [])
      ].map(row => row.textContent)
    ).toEqual(['beta-1', 'beta-2'])
    expect(globalThis.document.body.textContent).toContain('To pick up a draggable item, press the space bar')
  })

  it('keeps branch children clustered under non-sortable parents in profile Manual mode', () => {
    const parent = { ...makeSession('parent', 3000), profile: 'alpha' }
    const child = { ...makeSession('child', 2500), parent_session_id: 'parent', profile: 'alpha' } as SessionInfo
    const other = { ...makeSession('other', 2000), profile: 'alpha' }

    render(
      <SidebarSessionsSection
        activeSessionId={null}
        emptyState={<div>Empty</div>}
        groups={[{ id: 'alpha', label: 'Alpha', mode: 'profile', path: null, sessions: [parent, child, other] }]}
        label="Sessions"
        manualOrderIdsByProfile={{ alpha: ['other', 'parent'] }}
        onArchiveSession={noop}
        onDeleteSession={noop}
        onReorderProfileSessions={noop}
        onResumeSession={noop}
        onToggle={noop}
        onTogglePin={noop}
        onToggleUnread={noop}
        open
        pinned={false}
        sessions={[]}
      />
    )

    const rows = [...globalThis.document.querySelectorAll('[data-testid^="session-row-"]')]

    expect(rows.map(row => row.textContent)).toEqual(['other', 'parent', '└─ child'])
    expect(
      globalThis.document.querySelector('[data-testid="session-row-parent"]')?.getAttribute('data-reorderable')
    ).toBe('true')
    expect(
      globalThis.document.querySelector('[data-testid="session-row-child"]')?.getAttribute('data-reorderable')
    ).toBe('false')
    expect(
      globalThis.document.querySelector('[data-testid="session-row-child"]')?.getAttribute('data-branch-stem')
    ).toBe('└─ ')
  })

  it('keeps grouped rows static when Manual mode is not active', () => {
    const rows = [makeSession('alpha-1'), makeSession('alpha-2')]

    render(
      <SidebarSessionsSection
        activeSessionId={null}
        emptyState={<div>Empty</div>}
        groups={[{ id: 'alpha', label: 'Alpha', mode: 'profile', path: null, sessions: rows }]}
        label="Sessions"
        onArchiveSession={noop}
        onDeleteSession={noop}
        onResumeSession={noop}
        onToggle={noop}
        onTogglePin={noop}
        onToggleUnread={noop}
        open
        pinned={false}
        sessions={[]}
      />
    )

    expect(globalThis.document.querySelectorAll('[data-reorderable="true"]')).toHaveLength(0)
  })

  it('renders colliding bare ids from each profile using that profile order', () => {
    const alpha = [makeSession('shared'), makeSession('alpha-2')].map(session => ({ ...session, profile: 'alpha' }))
    const beta = [makeSession('shared'), makeSession('beta-2')].map(session => ({ ...session, profile: 'beta' }))

    render(
      <SidebarSessionsSection
        activeSessionId={null}
        emptyState={<div>Empty</div>}
        groups={[
          { id: 'alpha', label: 'Alpha', mode: 'profile', path: null, sessions: alpha },
          { id: 'beta', label: 'Beta', mode: 'profile', path: null, sessions: beta }
        ]}
        label="Sessions"
        manualOrderIdsByProfile={{
          alpha: ['shared', 'alpha-2'],
          beta: ['beta-2', 'shared']
        }}
        onArchiveSession={noop}
        onDeleteSession={noop}
        onReorderProfileSessions={noop}
        onResumeSession={noop}
        onToggle={noop}
        onTogglePin={noop}
        onToggleUnread={noop}
        open
        pinned={false}
        sessions={[]}
      />
    )

    const rowIds = (profile: string) =>
      [
        ...(globalThis.document
          .querySelector(`[data-testid="profile-group-${profile}"]`)
          ?.querySelectorAll('[data-testid^="session-row-"]') ?? [])
      ].map(row => row.textContent)

    expect(rowIds('alpha')).toEqual(['shared', 'alpha-2'])
    expect(rowIds('beta')).toEqual(['beta-2', 'shared'])
  })
})
