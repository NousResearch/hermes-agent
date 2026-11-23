// The row menu labels its archive verb from the clicked row (Archive or
// Unarchive), so the row itself must reach the handler. An archived search hit
// is in no loaded list, and a handler that re-derived the state from those
// lists ran Archive under an Unarchive label.
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { SessionInfo } from '@/hermes'

import { SidebarSessionsSection } from './sessions-section'

vi.mock('./session-row', () => ({
  SidebarSessionRow: ({ onArchive, session }: { onArchive: () => void; session: SessionInfo }) => (
    <button onClick={onArchive} type="button">
      {`archive-verb-${session.id}`}
    </button>
  )
}))

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      common: { cancel: 'Cancel' },
      desktop: {},
      sidebar: {
        dateDivider: {
          earlier: 'Earlier',
          earlierThisMonth: 'Earlier this month',
          lastMonth: 'Last month',
          lastWeek: 'Last week',
          older: 'Older',
          thisMonth: 'This month',
          thisWeek: 'This week',
          today: 'Today',
          yesterday: 'Yesterday'
        },
        nav: { 'new-session': 'New session' },
        noSessions: 'No sessions yet'
      }
    }
  })
}))

afterEach(cleanup)

const searchHit = {
  archived: true,
  id: 'hit-1',
  last_active: 1,
  message_count: 0,
  started_at: 1,
  title: 'Archived chat'
} as SessionInfo

describe('SidebarSessionsSection archive verb', () => {
  it('hands the clicked row, with its archived flag, to the archive handler', () => {
    const onArchiveSession = vi.fn()
    const noop = vi.fn()

    render(
      <SidebarSessionsSection
        activeSessionId={null}
        emptyState={null}
        label="Search"
        onArchiveSession={onArchiveSession}
        onDeleteSession={noop}
        onResumeSession={noop}
        onToggle={noop}
        onTogglePin={noop}
        onToggleUnread={noop}
        open
        pinned={false}
        sessions={[searchHit]}
      />
    )

    fireEvent.click(screen.getByText('archive-verb-hit-1'))

    expect(onArchiveSession).toHaveBeenCalledWith('hit-1', expect.objectContaining({ archived: true }))
  })
})
