import { act, cleanup, render } from '@testing-library/react'
import type * as React from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { SessionInfo } from '@/hermes'
import { createClientSessionState } from '@/lib/chat-runtime'
import { $sessions, $unreadFinishedSessionIds } from '@/store/session'
import { $sessionDotStateById, sessionStatusBucket } from '@/store/session-dot-state'
import { clearAllSessionStates, publishSessionState } from '@/store/session-states'
import { $subagentsBySession, upsertSubagent } from '@/store/subagents'

import { SidebarSessionsSection, VIRTUALIZE_THRESHOLD } from './sessions-section'
import type { VirtualSessionListProps } from './virtual-session-list'

afterEach(() => {
  cleanup()
  clearAllSessionStates()
  $sessions.set([])
  $unreadFinishedSessionIds.set([])
  $subagentsBySession.set({})
})

const statusDivider = vi.hoisted(() => ({ done: 'Done', working: 'Working' }))

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      sidebar: {
        statusDivider,
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
  SidebarSessionRow: ({ session }: { session: SessionInfo }) => (
    <div data-testid={`session-row-${session.id}`}>{session.id}</div>
  )
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
  it('keeps delegated work in Working until it settles, independently of unread state', () => {
    mockVirtualListPropsHistory.length = 0
    const sessions = generateSessions(VIRTUALIZE_THRESHOLD + 1)
    const storedId = sessions[0].id
    $sessions.set(sessions)
    const state = createClientSessionState(storedId)
    publishSessionState('runtime', { ...state, busy: true })

    render(
      <SidebarSessionsSection
        activeSessionId={null}
        emptyState={<div>Empty</div>}
        grouping="status"
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

    const groupOfSession = () => {
      let group = ''

      for (const row of mockVirtualListPropsHistory.at(-1)!.rows) {
        if (row.kind === 'divider') {
          group = row.key
        } else if (row.entry.session.id === storedId) {
          return group
        }
      }
    }

    expect(groupOfSession()).toBe('status:working')
    act(() => {
      upsertSubagent('runtime', { subagent_id: 'child', status: 'queued' }, true, 'subagent.start')
      publishSessionState('runtime', { ...state, busy: false })
    })
    expect(sessionStatusBucket($sessionDotStateById.get()[storedId])).toBe('working')
    expect(groupOfSession()).toBe('status:working')

    act(() => {
      upsertSubagent('runtime', { subagent_id: 'child', status: 'running' }, true, 'subagent.progress')
      $unreadFinishedSessionIds.set([storedId])
    })
    expect(groupOfSession()).toBe('status:working')
    act(() => $unreadFinishedSessionIds.set([]))
    expect(groupOfSession()).toBe('status:working')

    act(() => publishSessionState('runtime', { ...state, busy: true, needsInput: true }))
    expect($sessionDotStateById.get()[storedId]).toBe('needs-input')
    expect(groupOfSession()).toBe('status:working')
    act(() => publishSessionState('runtime', { ...state, busy: false }))
    expect(sessionStatusBucket($sessionDotStateById.get()[storedId])).toBe('working')
    expect(groupOfSession()).toBe('status:working')

    act(() => {
      upsertSubagent('runtime', { subagent_id: 'child', status: 'completed' }, true, 'subagent.complete')
      publishSessionState('runtime', { ...state, busy: true })
    })
    expect(groupOfSession()).toBe('status:working')
    act(() => {
      publishSessionState('runtime', { ...state, busy: false })
      $unreadFinishedSessionIds.set([storedId])
    })
    expect($sessionDotStateById.get()[storedId]).toBe('unread')
    expect(groupOfSession()).toBe('status:done')
    act(() => $unreadFinishedSessionIds.set([]))
    expect(groupOfSession()).toBe('status:done')
  })
})
