import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { SessionInfo } from '@/hermes'
import { I18nProvider } from '@/i18n'
import { $collapsedSessionDateGroups } from '@/store/session-date-group-collapse'

import { SidebarSessionsSection } from './sessions-section'

const NOW = new Date(2026, 5, 18, 12, 0, 0)

const at = (day: number, hour: number, minute = 0): number =>
  Math.floor(new Date(2026, 5, day, hour, minute).getTime() / 1000)

const session = (id: string, lastActive: number): SessionInfo =>
  ({
    id,
    last_active: lastActive,
    message_count: 1,
    source: 'cli',
    started_at: lastActive,
    title: id,
    tool_call_count: 0
  }) as SessionInfo

const renderSection = (sessions: SessionInfo[]) =>
  render(
    <I18nProvider configClient={null} initialLocale="en">
      <SidebarSessionsSection
        activeSessionId={null}
        dateGroupScope="test-scope"
        emptyState={null}
        grouping="date"
        label="Sessions"
        onArchiveSession={vi.fn()}
        onDeleteSession={vi.fn()}
        onResumeSession={vi.fn()}
        onToggle={vi.fn()}
        onTogglePin={vi.fn()}
        onToggleUnread={vi.fn()}
        open
        pinned={false}
        sessions={sessions}
      />
    </I18nProvider>
  )

describe('SidebarSessionsSection date group collapse', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    vi.setSystemTime(NOW)
    act(() => $collapsedSessionDateGroups.set({}))
  })

  afterEach(() => {
    cleanup()
    vi.useRealTimers()
    $collapsedSessionDateGroups.set({})
  })

  it('renders and persists the first Today group instead of an unlabelled recent head', () => {
    renderSection([
      session('Today session', at(18, 11)),
      session('Yesterday session', at(17, 10)),
      session('Last week session', at(12, 10))
    ])

    expect(screen.getByRole('button', { name: 'Collapse Today' })).not.toBeNull()
    expect(screen.getByRole('button', { name: 'Collapse Yesterday' })).not.toBeNull()
    expect(screen.getByRole('button', { name: /^Collapse Last week/ })).not.toBeNull()
    expect(screen.getByText('Today session')).not.toBeNull()
    expect(screen.getByText('Yesterday session')).not.toBeNull()

    fireEvent.click(screen.getByRole('button', { name: 'Collapse Today' }))

    expect(screen.queryByText('Today session')).toBeNull()
    expect(screen.getByText('Yesterday session')).not.toBeNull()
    expect(screen.getByText('Last week session')).not.toBeNull()
    expect(screen.getByRole('button', { name: 'Expand Today' })).not.toBeNull()
    expect($collapsedSessionDateGroups.get()).toEqual({ 'test-scope': ['day:2026-06-18'] })

    fireEvent.click(screen.getByRole('button', { name: 'Expand Today' }))

    expect(screen.getByText('Today session')).not.toBeNull()
    expect($collapsedSessionDateGroups.get()).toEqual({})
  })
})
