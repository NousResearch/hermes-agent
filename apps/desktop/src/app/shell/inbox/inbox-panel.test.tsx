import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { type InboxEntry, type InboxItem } from '@/store/inbox'

import { InboxPanel } from './inbox-panel'

vi.mock('@/i18n', () => ({
  translateNow: (key: string) => key,
  useI18n: () => ({
    t: Object.assign((key: string) => key, {
      ui: { search: { clear: 'Clear' } }
    })
  })
}))

vi.mock('@/app/overlays/panel', async importActual => {
  const actual = await importActual<Record<string, unknown>>()

  return {
    ...actual,
    Panel: ({ children, onClose }: { children: React.ReactNode; onClose: () => void }) => (
      <div data-testid="panel">
        <button aria-label="Close" onClick={onClose} type="button" />
        {children}
      </div>
    )
  }
})

const navigateSpy = vi.fn()

vi.mock('react-router', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  useNavigate: () => navigateSpy
}))

vi.mock('@/store/profile', () => ({ $activeGatewayProfile: { get: () => 'inbox-test-profile' } }))

vi.mock('@/store/session', () => ({
  setSelectedStoredSessionId: vi.fn()
}))

vi.mock('@/store/inbox', async importActual => ({
  ...(await importActual<Record<string, unknown>>()),
  refreshInbox: vi.fn()
}))

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

function makeItem(overrides: Partial<InboxItem> & { session_key: string }): InboxItem {
  return {
    cwd: '/work/project',
    goal: null,
    heartbeat: null,
    lanes: ['needs_you'],
    loop: null,
    pending_approval: null,
    pending_clarify: null,
    source: 'cli',
    title: 'Test session',
    ...overrides
  }
}

function makeEntry(overrides: Partial<InboxEntry> = {}): InboxEntry {
  return {
    capability: 'supported',
    error: null,
    loading: false,
    snapshot: {
      badge: 'none',
      counts: { needs_you: 0, running: 0, waiting: 0, scheduled: 0, total: 0 },
      coverage: {
        approval_scope: 'live gateway approval queue',
        clarify_scope: 'live open sessions only',
        connection_scope: 'active connection and profile only',
        errors: [],
        partial: false,
        profile: 'default',
        scanned_sessions: 0
      },
      items: []
    },
    ...overrides
  }
}

function renderPanel(inbox: InboxEntry, onClose = vi.fn()) {
  return render(
    <MemoryRouter>
      <InboxPanel inbox={inbox} onClose={onClose} />
    </MemoryRouter>
  )
}

describe('InboxPanel', () => {
  it('renders the panel header with title and coverage', () => {
    renderPanel(makeEntry())
    expect(screen.getByText('Agent Inbox')).toBeTruthy()
  })

  it('shows two sections: Needs you and Automation (not three tabs)', () => {
    renderPanel(makeEntry())
    expect(screen.getByText('Needs you')).toBeTruthy()
    expect(screen.getByText('Automation')).toBeTruthy()
    // Must NOT have a "Scheduled" standalone section
    const allButtons = screen.getAllByRole('button')
    const scheduledButtons = allButtons.filter(b => b.textContent?.includes('Scheduled') && !b.textContent?.includes('Automation'))
    expect(scheduledButtons).toHaveLength(0)
  })

  it('renders metadata-only pending_clarify with safe fixed text', () => {
    const item = makeItem({
      pending_clarify: { count: 2 },
      session_key: 'sess-clarify'
    })

    const entry = makeEntry({
      snapshot: {
        badge: 'amber',
        counts: { needs_you: 1, running: 0, waiting: 0, scheduled: 0, total: 1 },
        coverage: {
          approval_scope: '',
          clarify_scope: '',
          connection_scope: '',
          errors: [],
          partial: false,
          profile: 'default',
          scanned_sessions: 1
        },
        items: [item]
      }
    })

    renderPanel(entry)

    // Select the item to see its detail
    fireEvent.click(screen.getByText('Test session'))

    // The detail should show safe fixed text, NOT the question content
    expect(screen.getByText('2 questions waiting in this chat')).toBeTruthy()
    // Must NOT try to render a question string
    expect(screen.queryByText('undefined')).toBeNull()
  })

  it('renders metadata-only pending_clarify with count=1 (singular)', () => {
    const item = makeItem({
      pending_clarify: { count: 1 },
      session_key: 'sess-one-clarify'
    })

    const entry = makeEntry({
      snapshot: {
        badge: 'amber',
        counts: { needs_you: 1, running: 0, waiting: 0, scheduled: 0, total: 1 },
        coverage: {
          approval_scope: '',
          clarify_scope: '',
          connection_scope: '',
          errors: [],
          partial: false,
          profile: 'default',
          scanned_sessions: 1
        },
        items: [item]
      }
    })

    renderPanel(entry)
    fireEvent.click(screen.getByText('Test session'))
    expect(screen.getByText('1 question waiting in this chat')).toBeTruthy()
  })

  it('groups items by section: needs_you in Needs you, running/waiting/scheduled in Automation', () => {
    const needsItem = makeItem({ lanes: ['needs_you'], session_key: 'sess-needs', title: 'Needs item' })
    const runningItem = makeItem({ lanes: ['running'], session_key: 'sess-running', title: 'Running item' })
    const waitingItem = makeItem({ lanes: ['waiting'], session_key: 'sess-waiting', title: 'Waiting item' })
    const scheduledItem = makeItem({ lanes: ['scheduled'], session_key: 'sess-scheduled', title: 'Scheduled item' })

    const entry = makeEntry({
      snapshot: {
        badge: 'amber',
        counts: { needs_you: 1, running: 1, waiting: 1, scheduled: 1, total: 4 },
        coverage: {
          approval_scope: '',
          clarify_scope: '',
          connection_scope: '',
          errors: [],
          partial: false,
          profile: 'default',
          scanned_sessions: 4
        },
        items: [needsItem, runningItem, waitingItem, scheduledItem]
      }
    })

    renderPanel(entry)

    // Default section is "Needs you" — only needs item visible
    expect(screen.getByText('Needs item')).toBeTruthy()
    expect(screen.queryByText('Running item')).toBeNull()
    expect(screen.queryByText('Waiting item')).toBeNull()
    expect(screen.queryByText('Scheduled item')).toBeNull()

    // Switch to Automation section
    fireEvent.click(screen.getByText('Automation'))
    expect(screen.queryByText('Needs item')).toBeNull()
    expect(screen.getByText('Running item')).toBeTruthy()
    expect(screen.getByText('Waiting item')).toBeTruthy()
    expect(screen.getByText('Scheduled item')).toBeTruthy()
  })

  it('shows "All clear" for genuinely empty inbox', () => {
    renderPanel(makeEntry())
    expect(screen.getByText('All clear')).toBeTruthy()
  })

  it('shows "No results for filter" when search matches nothing', () => {
    const item = makeItem({ title: 'My project', session_key: 'sess-1' })

    const entry = makeEntry({
      snapshot: {
        badge: 'none',
        counts: { needs_you: 1, running: 0, waiting: 0, scheduled: 0, total: 1 },
        coverage: {
          approval_scope: '',
          clarify_scope: '',
          connection_scope: '',
          errors: [],
          partial: false,
          profile: 'default',
          scanned_sessions: 1
        },
        items: [item]
      }
    })

    renderPanel(entry)

    // Search for something that doesn't exist
    const searchInput = screen.getByPlaceholderText('Filter sessions…')
    fireEvent.change(searchInput, { target: { value: 'no-match-xyz' } })

    expect(screen.getByText('No results for filter')).toBeTruthy()
  })

  it('shows error banner but preserves usable rows when coverage has errors', () => {
    const usableItem = makeItem({ title: 'Still here', session_key: 'sess-ok' })

    const entry = makeEntry({
      snapshot: {
        badge: 'red',
        counts: { needs_you: 1, running: 0, waiting: 0, scheduled: 0, total: 1 },
        coverage: {
          approval_scope: '',
          clarify_scope: '',
          connection_scope: '',
          errors: ['sess-failed: snapshot read failed'],
          partial: true,
          profile: 'default',
          scanned_sessions: 2
        },
        items: [usableItem]
      }
    })

    renderPanel(entry)

    // Error banner is shown
    expect(screen.getByText('Partial read')).toBeTruthy()
    expect(screen.getByText('1 session snapshot could not be read. Counts below are incomplete.')).toBeTruthy()
    // Refresh button is visible for partial read state
    expect(screen.getByRole('button', { name: /refresh/i })).toBeTruthy()
    // Usable row is still visible
    expect(screen.getByText('Still here')).toBeTruthy()
  })

  it('shows "Incomplete data" for partial coverage with no rows', () => {
    const entry = makeEntry({
      snapshot: {
        badge: 'amber',
        counts: { needs_you: 0, running: 0, waiting: 0, scheduled: 0, total: 0 },
        coverage: {
          approval_scope: '',
          clarify_scope: '',
          connection_scope: '',
          errors: [],
          partial: true,
          profile: 'default',
          scanned_sessions: 0
        },
        items: []
      }
    })

    renderPanel(entry)
    expect(screen.getByText('Incomplete data')).toBeTruthy()
    expect(screen.getByText('Some sessions could not be read. Counts may be incomplete.')).toBeTruthy()
    // Refresh button is visible for incomplete data state
    expect(screen.getByRole('button', { name: /refresh/i })).toBeTruthy()
  })

  it('shows disconnected state with retry button', () => {
    const entry = makeEntry({
      error: 'gateway reset',
      snapshot: null
    })

    renderPanel(entry)

    expect(screen.getByText('Disconnected')).toBeTruthy()
    expect(screen.getByText('gateway reset')).toBeTruthy()
    expect(screen.getByRole('button', { name: /retry/i })).toBeTruthy()
  })

  it('retry button calls refreshInbox', async () => {
    const { refreshInbox } = await import('@/store/inbox')
    const entry = makeEntry({ error: 'gateway reset', snapshot: null })
    renderPanel(entry)

    fireEvent.click(screen.getByRole('button', { name: /retry/i }))
    expect(refreshInbox).toHaveBeenCalled()
  })

  it('refresh button in partial read state calls refreshInbox', async () => {
    const { refreshInbox } = await import('@/store/inbox')

    const entry = makeEntry({
      snapshot: {
        badge: 'red',
        counts: { needs_you: 0, running: 0, waiting: 0, scheduled: 0, total: 0 },
        coverage: {
          approval_scope: '',
          clarify_scope: '',
          connection_scope: '',
          errors: ['sess-failed: snapshot read failed'],
          partial: true,
          profile: 'default',
          scanned_sessions: 1
        },
        items: []
      }
    })

    renderPanel(entry)

    fireEvent.click(screen.getByRole('button', { name: /refresh/i }))
    expect(refreshInbox).toHaveBeenCalled()
  })

  it('shows unsupported state for old gateway', () => {
    const entry = makeEntry({ capability: 'unsupported' })
    renderPanel(entry)
    expect(screen.getByText('Inbox not supported')).toBeTruthy()
    expect(screen.getByText(/does not expose the inbox aggregation/)).toBeTruthy()
  })

  it('shows loading state while fetching', () => {
    const entry = makeEntry({ snapshot: null })
    renderPanel(entry)
    expect(screen.getByText('Loading inbox…')).toBeTruthy()
  })

  it('closing the panel calls onClose', () => {
    const onClose = vi.fn()
    renderPanel(makeEntry(), onClose)

    fireEvent.click(screen.getByRole('button', { name: 'Close' }))
    expect(onClose).toHaveBeenCalledOnce()
  })

  it('selecting a row and clicking Open session navigates to the session route', () => {
    const item = makeItem({ title: 'Nav test', session_key: 'sess-nav' })

    const entry = makeEntry({
      snapshot: {
        badge: 'none',
        counts: { needs_you: 1, running: 0, waiting: 0, scheduled: 0, total: 1 },
        coverage: {
          approval_scope: '',
          clarify_scope: '',
          connection_scope: '',
          errors: [],
          partial: false,
          profile: 'default',
          scanned_sessions: 1
        },
        items: [item]
      }
    })

    renderPanel(entry)

    // Select the item
    fireEvent.click(screen.getByText('Nav test'))
    // Click "Open session" button in detail
    fireEvent.click(screen.getByText('Open session'))

    expect(navigateSpy).toHaveBeenCalledWith(expect.stringContaining('sess-nav'))
  })

  it('renders pending_approval with description', () => {
    const item = makeItem({
      pending_approval: { command_redacted: true, count: 1, description: 'run deploy' },
      session_key: 'sess-approval'
    })

    const entry = makeEntry({
      snapshot: {
        badge: 'amber',
        counts: { needs_you: 1, running: 0, waiting: 0, scheduled: 0, total: 1 },
        coverage: {
          approval_scope: '',
          clarify_scope: '',
          connection_scope: '',
          errors: [],
          partial: false,
          profile: 'default',
          scanned_sessions: 1
        },
        items: [item]
      }
    })

    renderPanel(entry)
    fireEvent.click(screen.getByText('Test session'))
    expect(screen.getByText('run deploy')).toBeTruthy()
  })

  it('renders goal and heartbeat metadata in detail pane', () => {
    const item = makeItem({
      goal: { status: 'active', title: 'Ship inbox' },
      heartbeat: { status: 'active', prompt: 'Poll' },
      session_key: 'sess-meta'
    })

    const entry = makeEntry({
      snapshot: {
        badge: 'none',
        counts: { needs_you: 1, running: 0, waiting: 0, scheduled: 0, total: 1 },
        coverage: {
          approval_scope: '',
          clarify_scope: '',
          connection_scope: '',
          errors: [],
          partial: false,
          profile: 'default',
          scanned_sessions: 1
        },
        items: [item]
      }
    })

    renderPanel(entry)
    fireEvent.click(screen.getByText('Test session'))
    expect(screen.getByText('Goal')).toBeTruthy()
    expect(screen.getByText('Ship inbox')).toBeTruthy()
    expect(screen.getByText('Heartbeat')).toBeTruthy()
  })

  it('hides detail pane when filtered visible list is empty (no "Select a session" placeholder)', () => {
    const item = makeItem({ title: 'My project', session_key: 'sess-1' })

    const entry = makeEntry({
      snapshot: {
        badge: 'none',
        counts: { needs_you: 1, running: 0, waiting: 0, scheduled: 0, total: 1 },
        coverage: {
          approval_scope: '',
          clarify_scope: '',
          connection_scope: '',
          errors: [],
          partial: false,
          profile: 'default',
          scanned_sessions: 1
        },
        items: [item]
      }
    })

    renderPanel(entry)

    // Search for something that doesn't exist — list becomes empty
    const searchInput = screen.getByPlaceholderText('Filter sessions…')
    fireEvent.change(searchInput, { target: { value: 'no-match-xyz' } })

    // Empty state should be shown in the list area
    expect(screen.getByText('No results for filter')).toBeTruthy()

    // Detail pane should NOT show "Select a session" — it's hidden when list is empty
    expect(screen.queryByText('Select a session')).toBeNull()
  })

  it('shows detail pane when filtered list has items but none selected', () => {
    const item = makeItem({ title: 'My project', session_key: 'sess-1' })

    const entry = makeEntry({
      snapshot: {
        badge: 'none',
        counts: { needs_you: 1, running: 0, waiting: 0, scheduled: 0, total: 1 },
        coverage: {
          approval_scope: '',
          clarify_scope: '',
          connection_scope: '',
          errors: [],
          partial: false,
          profile: 'default',
          scanned_sessions: 1
        },
        items: [item]
      }
    })

    renderPanel(entry)

    // Items are visible but none selected — detail pane shows placeholder
    expect(screen.getByText('My project')).toBeTruthy()
    expect(screen.getByText('Select a session')).toBeTruthy()
  })

  it('subtitle shows "Profile: <name> · This connection only"', () => {
    const entry = makeEntry({
      snapshot: {
        badge: 'none',
        counts: { needs_you: 0, running: 0, waiting: 0, scheduled: 0, total: 0 },
        coverage: {
          approval_scope: '',
          clarify_scope: '',
          connection_scope: 'active connection and profile only',
          errors: [],
          partial: false,
          profile: 'my-profile',
          scanned_sessions: 0
        },
        items: []
      }
    })

    renderPanel(entry)
    expect(screen.getByText('Profile: my-profile · This connection only')).toBeTruthy()
  })

  it('hides detail pane when section is genuinely empty (All clear)', () => {
    renderPanel(makeEntry())

    // All clear is shown in the list area
    expect(screen.getByText('All clear')).toBeTruthy()

    // Detail pane is not shown when list is empty
    expect(screen.queryByText('Select a session')).toBeNull()
  })
})
