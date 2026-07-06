import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $attentionSessionIds, $selectedStoredSessionId } from '@/store/session'
import { $workstreamFilter } from '@/store/workstream-filter'
import { $workstreamMetadata } from '@/store/workstream-metadata'
import type { SessionInfo } from '@/types/hermes'

import { SIDEBAR_GROUP_PAGE } from './projects/model'
import { SidebarSessionsSection, VIRTUALIZE_THRESHOLD } from './sessions-section'

const noop = () => undefined

const session = (id: string, title = id): SessionInfo =>
  ({
    archived: false,
    cwd: null,
    ended_at: null,
    id,
    input_tokens: 0,
    is_active: false,
    last_active: 0,
    message_count: 1,
    model: null,
    output_tokens: 0,
    preview: null,
    source: null,
    started_at: 0,
    title,
    tool_call_count: 0
  }) as SessionInfo

const renderSection = (sessions: SessionInfo[]) =>
  render(
    <SidebarSessionsSection
      activeSessionId={null}
      emptyState={<div>No rows</div>}
      label="Sessions"
      onArchiveSession={noop}
      onDeleteSession={noop}
      onResumeSession={noop}
      onToggle={noop}
      onTogglePin={noop}
      open
      pinned={false}
      sessions={sessions}
      workingSessionIdSet={new Set()}
    />
  )

describe('SidebarSessionsSection workstream filtering', () => {
  beforeEach(() => {
    $workstreamFilter.set('all')
    $attentionSessionIds.set([])
    $selectedStoredSessionId.set(null)
    $workstreamMetadata.set({})
  })

  afterEach(() => cleanup())

  it('filters closed and safe-delete rows separately', () => {
    $workstreamFilter.set('safe-delete')
    $workstreamMetadata.set({
      closed: { lifecycle: 'closed', updatedAt: 1 },
      safe: { lifecycle: 'safe_delete', updatedAt: 2 }
    })

    renderSection([session('active', 'Active'), session('closed', 'Closed'), session('safe', 'Safe')])

    expect(screen.queryByText('Safe')).not.toBeNull()
    expect(screen.queryByText('Closed')).toBeNull()
    expect(screen.queryByText('Active')).toBeNull()
  })

  it('filters before rows enter the virtualized list', () => {
    $workstreamFilter.set('closed')
    $workstreamMetadata.set({ closed: { lifecycle: 'closed', updatedAt: 1 } })

    const sessions = Array.from({ length: VIRTUALIZE_THRESHOLD + 1 }, (_, index) =>
      session(index === VIRTUALIZE_THRESHOLD ? 'closed' : `active-${index}`, index === VIRTUALIZE_THRESHOLD ? 'Closed row' : `Active ${index}`)
    )

    renderSection(sessions)

    expect(screen.queryByText('Closed row')).not.toBeNull()
    expect(screen.queryByText('Active 0')).toBeNull()
  })

  it('does not persist a reorder subset while a non-all workstream filter is active', () => {
    const onReorderSessions = vi.fn()
    $workstreamFilter.set('closed')
    $workstreamMetadata.set({ closed: { lifecycle: 'closed', updatedAt: 1 } })

    const { container } = render(
      <SidebarSessionsSection
        activeSessionId={null}
        emptyState={<div>No rows</div>}
        label="Sessions"
        onArchiveSession={noop}
        onDeleteSession={noop}
        onReorderSessions={onReorderSessions}
        onResumeSession={noop}
        onToggle={noop}
        onTogglePin={noop}
        open
        pinned={false}
        sessions={[session('closed', 'Closed'), session('hidden', 'Hidden')]}
        sortable
        workingSessionIdSet={new Set()}
      />
    )

    expect(container.querySelector('[data-reorder-handle]')).toBeNull()
    expect(screen.queryByText('Closed')).not.toBeNull()
  })

  it('filters selected stored sessions against their live runtime activity', () => {
    $workstreamFilter.set('blocked')
    $attentionSessionIds.set(['runtime-1'])
    $selectedStoredSessionId.set('stored-1')

    render(
      <SidebarSessionsSection
        activeRuntimeSessionId="runtime-1"
        activeSessionId="stored-1"
        emptyState={<div>No rows</div>}
        label="Sessions"
        onArchiveSession={noop}
        onDeleteSession={noop}
        onResumeSession={noop}
        onToggle={noop}
        onTogglePin={noop}
        open
        pinned={false}
        sessions={[session('stored-1', 'Stored'), session('other', 'Other')]}
        workingSessionIdSet={new Set()}
      />
    )

    expect(screen.queryByText('Stored')).not.toBeNull()
    expect(screen.queryByText('Other')).toBeNull()
  })

  it('marks only rendered grouped rows as workstream navigation targets', () => {
    const visibleSessions = Array.from({ length: SIDEBAR_GROUP_PAGE }, (_, index) => session(`visible-${index}`))
    const hiddenSession = session('hidden-behind-show-more')

    const { container } = render(
      <SidebarSessionsSection
        activeSessionId={null}
        emptyState={<div>No rows</div>}
        groups={[
          {
            id: 'group-1',
            label: 'Group',
            path: null,
            sessions: [...visibleSessions, hiddenSession]
          }
        ]}
        label="Sessions"
        onArchiveSession={noop}
        onDeleteSession={noop}
        onResumeSession={noop}
        onToggle={noop}
        onTogglePin={noop}
        open
        pinned={false}
        sessions={[]}
        workingSessionIdSet={new Set()}
      />
    )

    const ids = [...container.querySelectorAll<HTMLElement>('[data-workstream-session-id]')].map(
      row => row.dataset.workstreamSessionId
    )

    expect(ids).toEqual(visibleSessions.map(item => item.id))
    expect(screen.queryByText(hiddenSession.id)).toBeNull()
  })
})
