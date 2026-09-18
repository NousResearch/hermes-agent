import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
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

vi.mock('@/store/profile', () => ({
  $activeGatewayProfile: {
    get: vi.fn(() => 'inbox-test-profile'),
    listen: vi.fn(() => () => {})
  }
}))

vi.mock('@/store/session', () => ({
  setSelectedStoredSessionId: vi.fn(),
  $connection: {
    get: vi.fn(() => null),
    listen: vi.fn(() => () => {})
  }
}))

vi.mock('@/store/gateway', () => {
  let _value: unknown = null
  const _listeners: Array<(v: unknown) => void> = []
  const gw = {
    get() { return _value },
    set(v: unknown) { _value = v; _listeners.forEach(cb => cb(v)) },
    listen(cb: (v: unknown) => void) {
      _listeners.push(cb)
      return () => { const i = _listeners.indexOf(cb); if (i >= 0) _listeners.splice(i, 1) }
    },
    get value() { return _value }
  }
  return { $gateway: gw }
})

vi.mock('@/store/inbox', async importActual => ({
  ...(await importActual<Record<string, unknown>>()),
  refreshInbox: vi.fn(),
  fetchInboxRequestDetails: vi.fn().mockResolvedValue(null)
}))

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

function makeItem(overrides: Partial<InboxItem> & { session_key: string }): InboxItem {
  const { session_key, ...rest } = overrides

  return {
    background_task_count: 0,
    background_task_count_unavailable: false,
    categories: [],
    cwd: '/work/project',
    goal: null,
    heartbeat: null,
    lanes: ['needs_you'],
    loop: null,
    pending_approval: null,
    pending_clarify: null,
    session_key,
    source: 'cli',
    subagent_count: 0,
    subagent_count_unavailable: false,
    title: 'Test session',
    ...rest
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

function clickRow(title: string) {
  const matches = screen.getAllByText(title)
  const button = matches.find(el => el.closest('button'))?.closest('button')
  fireEvent.click(button!)
}

describe('InboxPanel', () => {
  it('renders the panel header with title and coverage', () => {
    renderPanel(makeEntry())
    expect(screen.getByText('Agent Inbox')).toBeTruthy()
  })

  it('renders left navigation with category items', () => {
    renderPanel(makeEntry())
    expect(screen.getByText('Needs attention')).toBeTruthy()
    const allSessionsButtons = screen.getAllByText('All sessions')
    expect(allSessionsButtons.length).toBeGreaterThanOrEqual(1)
    expect(screen.getByText('Goals')).toBeTruthy()
    expect(screen.getByText('Loops')).toBeTruthy()
    expect(screen.getByText('Heartbeats')).toBeTruthy()
    expect(screen.getByText('Background tasks')).toBeTruthy()
    expect(screen.getByText('Subagents')).toBeTruthy()
  })

  it('hides "Other" nav when no items have unrecognized categories', () => {
    renderPanel(makeEntry())
    expect(screen.queryByText('Other')).toBeNull()
  })

  it('shows "Other" nav when items have unrecognized categories', () => {
    const item = makeItem({ categories: ['unknown-cat'], session_key: 'sess-other' })

    const entry = makeEntry({
      snapshot: {
        badge: 'none',
        counts: { needs_you: 1, running: 0, waiting: 0, scheduled: 0, total: 1 },
        coverage: { approval_scope: '', clarify_scope: '', connection_scope: '', errors: [], partial: false, profile: 'default', scanned_sessions: 1 },
        items: [item]
      }
    })

    renderPanel(entry)
    expect(screen.getByText('Other')).toBeTruthy()
  })

  it('shows needs attention count in nav', () => {
    const item = makeItem({ lanes: ['needs_you'], session_key: 'sess-needs' })

    const entry = makeEntry({
      snapshot: {
        badge: 'amber',
        counts: { needs_you: 1, running: 0, waiting: 0, scheduled: 0, total: 1 },
        coverage: { approval_scope: '', clarify_scope: '', connection_scope: '', errors: [], partial: false, profile: 'default', scanned_sessions: 1 },
        items: [item]
      }
    })

    renderPanel(entry)
    const needsBtn = screen.getByText('Needs attention').closest('button')
    expect(needsBtn?.textContent).toContain('1')
  })

  it('shows category counts in nav', () => {
    const goalItem = makeItem({ categories: ['goals'], session_key: 'sess-goal' })

    const entry = makeEntry({
      snapshot: {
        badge: 'none',
        counts: { needs_you: 0, running: 1, waiting: 0, scheduled: 0, total: 1 },
        coverage: { approval_scope: '', clarify_scope: '', connection_scope: '', errors: [], partial: false, profile: 'default', scanned_sessions: 1 },
        items: [goalItem]
      }
    })

    renderPanel(entry)
    const goalsBtn = screen.getByText('Goals').closest('button')
    expect(goalsBtn?.textContent).toContain('1')
  })

  it('shows search field with scope toggle', () => {
    renderPanel(makeEntry())
    expect(screen.getByPlaceholderText('Search title, key, or path…')).toBeTruthy()
    expect(screen.getByText('This section')).toBeTruthy()
    expect(screen.getAllByText('All sessions').length).toBeGreaterThanOrEqual(1)
  })

  it('shows "All clear" for genuinely empty inbox', () => {
    renderPanel(makeEntry())
    expect(screen.getByText('All clear')).toBeTruthy()
  })

  it('shows "No results" when search matches nothing', () => {
    const item = makeItem({ title: 'My project', session_key: 'sess-1' })

    const entry = makeEntry({
      snapshot: {
        badge: 'none',
        counts: { needs_you: 1, running: 0, waiting: 0, scheduled: 0, total: 1 },
        coverage: { approval_scope: '', clarify_scope: '', connection_scope: '', errors: [], partial: false, profile: 'default', scanned_sessions: 1 },
        items: [item]
      }
    })

    renderPanel(entry)
    const searchInput = screen.getByPlaceholderText('Search title, key, or path…')
    fireEvent.change(searchInput, { target: { value: 'no-match-xyz' } })
    expect(screen.getByText('No results')).toBeTruthy()
  })

  it('shows session rows with title and meta', () => {
    const item = makeItem({ title: 'Deploy session', session_key: 'sess-deploy', subagent_count: 3 })

    const entry = makeEntry({
      snapshot: {
        badge: 'none',
        counts: { needs_you: 1, running: 0, waiting: 0, scheduled: 0, total: 1 },
        coverage: { approval_scope: '', clarify_scope: '', connection_scope: '', errors: [], partial: false, profile: 'default', scanned_sessions: 1 },
        items: [item]
      }
    })

    renderPanel(entry)
    expect(screen.getByText('Deploy session')).toBeTruthy()
  })

  it('expands session row on click and shows inline detail', () => {
    const item = makeItem({ title: 'Expand me', session_key: 'sess-expand' })

    const entry = makeEntry({
      snapshot: {
        badge: 'none',
        counts: { needs_you: 1, running: 0, waiting: 0, scheduled: 0, total: 1 },
        coverage: { approval_scope: '', clarify_scope: '', connection_scope: '', errors: [], partial: false, profile: 'default', scanned_sessions: 1 },
        items: [item]
      }
    })

    renderPanel(entry)
    clickRow('Expand me')
    expect(screen.getByText('Title')).toBeTruthy()
    expect(screen.getAllByText('Expand me').length).toBeGreaterThanOrEqual(1)
  })

  it('collapses expanded row on second click', () => {
    const item = makeItem({ title: 'Toggle', session_key: 'sess-toggle' })

    const entry = makeEntry({
      snapshot: {
        badge: 'none',
        counts: { needs_you: 1, running: 0, waiting: 0, scheduled: 0, total: 1 },
        coverage: { approval_scope: '', clarify_scope: '', connection_scope: '', errors: [], partial: false, profile: 'default', scanned_sessions: 1 },
        items: [item]
      }
    })

    renderPanel(entry)
    clickRow('Toggle')
    expect(screen.getByText('Title')).toBeTruthy()
    clickRow('Toggle')
    expect(screen.queryByText('Title')).toBeNull()
  })

  it('shows subagent and bg task counts in inline detail', () => {
    const item = makeItem({
      background_task_count: 5,
      background_task_count_unavailable: true,
      session_key: 'sess-counts',
      subagent_count: 3,
      subagent_count_unavailable: false,
      title: 'Counted session'
    })

    const entry = makeEntry({
      snapshot: {
        badge: 'none',
        counts: { needs_you: 1, running: 0, waiting: 0, scheduled: 0, total: 1 },
        coverage: { approval_scope: '', clarify_scope: '', connection_scope: '', errors: [], partial: false, profile: 'default', scanned_sessions: 1 },
        items: [item]
      }
    })

    renderPanel(entry)
    clickRow('Counted session')
    expect(screen.getAllByText('Subagents').length).toBeGreaterThanOrEqual(2)
    expect(screen.getByText('3')).toBeTruthy()
    expect(screen.getAllByText('BG tasks').length).toBeGreaterThanOrEqual(1)
    expect(screen.getByText('5+ (unavailable)')).toBeTruthy()
  })

  it('shows categories in inline detail when present', () => {
    const item = makeItem({ categories: ['goals', 'loops'], session_key: 'sess-cats', title: 'Categorized' })

    const entry = makeEntry({
      snapshot: {
        badge: 'none',
        counts: { needs_you: 1, running: 0, waiting: 0, scheduled: 0, total: 1 },
        coverage: { approval_scope: '', clarify_scope: '', connection_scope: '', errors: [], partial: false, profile: 'default', scanned_sessions: 1 },
        items: [item]
      }
    })

    renderPanel(entry)
    clickRow('Categorized')
    expect(screen.getByText('Categories')).toBeTruthy()
    expect(screen.getByText('goals, loops')).toBeTruthy()
  })

  it('shows error banner but preserves usable rows', () => {
    const usableItem = makeItem({ title: 'Still here', session_key: 'sess-ok' })

    const entry = makeEntry({
      snapshot: {
        badge: 'red',
        counts: { needs_you: 1, running: 0, waiting: 0, scheduled: 0, total: 1 },
        coverage: { approval_scope: '', clarify_scope: '', connection_scope: '', errors: ['sess-failed: read failed'], partial: true, profile: 'default', scanned_sessions: 2 },
        items: [usableItem]
      }
    })

    renderPanel(entry)
    expect(screen.getByText('Partial read')).toBeTruthy()
    expect(screen.getByText('Still here')).toBeTruthy()
  })

  it('shows disconnected state with retry button', () => {
    const entry = makeEntry({ error: 'gateway reset', snapshot: null })
    renderPanel(entry)
    expect(screen.getByText('Disconnected')).toBeTruthy()
    expect(screen.getByText('gateway reset')).toBeTruthy()
    expect(screen.getByRole('button', { name: /retry/i })).toBeTruthy()
  })

  it('shows unsupported state for old gateway', () => {
    const entry = makeEntry({ capability: 'unsupported' })
    renderPanel(entry)
    expect(screen.getByText('Inbox not supported')).toBeTruthy()
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

  it('shows context button in detail pane', () => {
    const item = makeItem({ session_key: 'sess-ctx', title: 'Context test' })

    const entry = makeEntry({
      snapshot: {
        badge: 'none',
        counts: { needs_you: 1, running: 0, waiting: 0, scheduled: 0, total: 1 },
        coverage: { approval_scope: '', clarify_scope: '', connection_scope: '', errors: [], partial: false, profile: 'default', scanned_sessions: 1 },
        items: [item]
      }
    })

    renderPanel(entry)
    clickRow('Context test')
    expect(screen.getByText('Open chat for context')).toBeTruthy()
  })

  it('goal and heartbeat metadata shown in inline detail', () => {
    const item = makeItem({
      goal: { status: 'active', title: 'Ship inbox' },
      heartbeat: { status: 'active' },
      session_key: 'sess-meta',
      title: 'Meta session'
    })

    const entry = makeEntry({
      snapshot: {
        badge: 'none',
        counts: { needs_you: 1, running: 0, waiting: 0, scheduled: 0, total: 1 },
        coverage: { approval_scope: '', clarify_scope: '', connection_scope: '', errors: [], partial: false, profile: 'default', scanned_sessions: 1 },
        items: [item]
      }
    })

    renderPanel(entry)
    clickRow('Meta session')
    expect(screen.getByText('Goal')).toBeTruthy()
    expect(screen.getByText('Ship inbox')).toBeTruthy()
    expect(screen.getByText('Heartbeat')).toBeTruthy()
  })

  it('hides inline detail when filtered list is empty', () => {
    const item = makeItem({ title: 'My project', session_key: 'sess-1' })

    const entry = makeEntry({
      snapshot: {
        badge: 'none',
        counts: { needs_you: 1, running: 0, waiting: 0, scheduled: 0, total: 1 },
        coverage: { approval_scope: '', clarify_scope: '', connection_scope: '', errors: [], partial: false, profile: 'default', scanned_sessions: 1 },
        items: [item]
      }
    })

    renderPanel(entry)
    const searchInput = screen.getByPlaceholderText('Search title, key, or path…')
    fireEvent.change(searchInput, { target: { value: 'no-match-xyz' } })
    expect(screen.getByText('No results')).toBeTruthy()
    expect(screen.queryByText('Title')).toBeNull()
  })

  it('subtitle shows profile and scanned sessions', () => {
    const entry = makeEntry({
      snapshot: {
        badge: 'none',
        counts: { needs_you: 0, running: 0, waiting: 0, scheduled: 0, total: 0 },
        coverage: { approval_scope: '', clarify_scope: '', connection_scope: '', errors: [], partial: false, profile: 'my-profile', scanned_sessions: 5 },
        items: []
      }
    })

    renderPanel(entry)
    expect(screen.getByText('Profile: my-profile · 5 sessions scanned')).toBeTruthy()
  })

  it('hides detail pane when section is genuinely empty (All clear)', () => {
    renderPanel(makeEntry())
    expect(screen.getByText('All clear')).toBeTruthy()
    expect(screen.queryByText('Select a session')).toBeNull()
  })

  it('switches category via nav buttons', () => {
    const goalItem = makeItem({ categories: ['goals'], session_key: 'sess-goal', title: 'Goal item' })
    const loopItem = makeItem({ categories: ['loops'], lanes: ['running'], session_key: 'sess-loop', title: 'Loop item' })

    const entry = makeEntry({
      snapshot: {
        badge: 'none',
        counts: { needs_you: 0, running: 1, waiting: 0, scheduled: 0, total: 2 },
        coverage: { approval_scope: '', clarify_scope: '', connection_scope: '', errors: [], partial: false, profile: 'default', scanned_sessions: 2 },
        items: [goalItem, loopItem]
      }
    })

    renderPanel(entry)
    expect(screen.getByText('Goal item')).toBeTruthy()
    expect(screen.getByText('Loop item')).toBeTruthy()
    fireEvent.click(screen.getByText('Goals'))
    expect(screen.getByText('Goal item')).toBeTruthy()
    expect(screen.queryByText('Loop item')).toBeNull()
    fireEvent.click(screen.getByText('Loops'))
    expect(screen.queryByText('Goal item')).toBeNull()
    expect(screen.getByText('Loop item')).toBeTruthy()
  })

  it('needs attention filter shows only items with needs_you lane', () => {
    const needsItem = makeItem({ lanes: ['needs_you'], session_key: 'sess-needs', title: 'Needs item' })
    const runningItem = makeItem({ lanes: ['running'], session_key: 'sess-running', title: 'Running item' })

    const entry = makeEntry({
      snapshot: {
        badge: 'amber',
        counts: { needs_you: 1, running: 1, waiting: 0, scheduled: 0, total: 2 },
        coverage: { approval_scope: '', clarify_scope: '', connection_scope: '', errors: [], partial: false, profile: 'default', scanned_sessions: 2 },
        items: [needsItem, runningItem]
      }
    })

    renderPanel(entry)
    expect(screen.getByText('Needs item')).toBeTruthy()
    expect(screen.getByText('Running item')).toBeTruthy()
    fireEvent.click(screen.getByText('Needs attention'))
    expect(screen.getByText('Needs item')).toBeTruthy()
    expect(screen.queryByText('Running item')).toBeNull()
  })

  it('needs attention filter is distinct from All category', () => {
    const goalItem = makeItem({ categories: ['goals'], lanes: ['needs_you'], session_key: 'sess-goal', title: 'Goal needs' })
    const goalOnly = makeItem({ categories: ['goals'], lanes: ['running'], session_key: 'sess-goal2', title: 'Goal running' })

    const entry = makeEntry({
      snapshot: {
        badge: 'amber',
        counts: { needs_you: 1, running: 1, waiting: 0, scheduled: 0, total: 2 },
        coverage: { approval_scope: '', clarify_scope: '', connection_scope: '', errors: [], partial: false, profile: 'default', scanned_sessions: 2 },
        items: [goalItem, goalOnly]
      }
    })

    renderPanel(entry)
    fireEvent.click(screen.getByText('Goals'))
    expect(screen.getByText('Goal needs')).toBeTruthy()
    expect(screen.getByText('Goal running')).toBeTruthy()
    fireEvent.click(screen.getByText('Needs attention'))
    expect(screen.getByText('Goal needs')).toBeTruthy()
    expect(screen.queryByText('Goal running')).toBeNull()
  })

  it('expanded row shows detail inline below the row with aria-expanded', async () => {
    const item = makeItem({ title: 'Inline detail', session_key: 'sess-inline' })

    const entry = makeEntry({
      snapshot: {
        badge: 'none',
        counts: { needs_you: 1, running: 0, waiting: 0, scheduled: 0, total: 1 },
        coverage: { approval_scope: '', clarify_scope: '', connection_scope: '', errors: [], partial: false, profile: 'default', scanned_sessions: 1 },
        items: [item]
      }
    })

    renderPanel(entry)
    const row = screen.getByText('Inline detail').closest('[data-panel-row]')
    expect(row).toBeTruthy()
    expect(row?.getAttribute('aria-expanded')).toBe('false')
    clickRow('Inline detail')
    expect(screen.getByText('Title')).toBeTruthy()
    expect(row?.getAttribute('aria-expanded')).toBe('true')
  })

  it('subagent and background task badges use readable labels', () => {
    const item = makeItem({
      background_task_count: 2,
      background_task_count_unavailable: false,
      session_key: 'sess-badges',
      subagent_count: 3,
      subagent_count_unavailable: false,
      title: 'Badge test'
    })

    const entry = makeEntry({
      snapshot: {
        badge: 'none',
        counts: { needs_you: 1, running: 0, waiting: 0, scheduled: 0, total: 1 },
        coverage: { approval_scope: '', clarify_scope: '', connection_scope: '', errors: [], partial: false, profile: 'default', scanned_sessions: 1 },
        items: [item]
      }
    })

    renderPanel(entry)
    expect(screen.getByText(/3 subagents?/)).toBeTruthy()
    expect(screen.getByText(/2 background tasks?/)).toBeTruthy()
  })

  it('unavailable subagent badge shows unavailable text', () => {
    const item = makeItem({
      background_task_count: 0,
      background_task_count_unavailable: true,
      session_key: 'sess-unavail',
      subagent_count: 0,
      subagent_count_unavailable: true,
      title: 'Unavailable test'
    })

    const entry = makeEntry({
      snapshot: {
        badge: 'none',
        counts: { needs_you: 1, running: 0, waiting: 0, scheduled: 0, total: 1 },
        coverage: { approval_scope: '', clarify_scope: '', connection_scope: '', errors: [], partial: false, profile: 'default', scanned_sessions: 1 },
        items: [item]
      }
    })

    renderPanel(entry)
    expect(screen.getByText(/Subagents unavailable/)).toBeTruthy()
    expect(screen.getByText(/Background tasks unavailable/)).toBeTruthy()
  })

  // ── Scope & global-search regression tests ──────────────────────────────────

  it('global search under needs-attention includes ordinary matching sessions', () => {
    const needsItem = makeItem({ lanes: ['needs_you'], session_key: 'sess-needs', title: 'Needs alpha' })
    const runningItem = makeItem({ lanes: ['running'], session_key: 'sess-running', title: 'Running beta' })

    const entry = makeEntry({
      snapshot: {
        badge: 'amber',
        counts: { needs_you: 1, running: 1, waiting: 0, scheduled: 0, total: 2 },
        coverage: { approval_scope: '', clarify_scope: '', connection_scope: '', errors: [], partial: false, profile: 'default', scanned_sessions: 2 },
        items: [needsItem, runningItem]
      }
    })

    renderPanel(entry)

    // Activate needs-attention filter
    fireEvent.click(screen.getByText('Needs attention'))
    expect(screen.getByText('Needs alpha')).toBeTruthy()
    expect(screen.queryByText('Running beta')).toBeNull()

    // Switch to global search scope — find the "All sessions" toggle in the segmented control
    const allSessionButtons = screen.getAllByText('All sessions')
    // The segmented control renders radio-style buttons; pick the one not inside the nav
    const toggle = allSessionButtons.find(el => !el.closest('[class*="min-h-0"]'))
      ?? allSessionButtons[allSessionButtons.length - 1]
    fireEvent.click(toggle)

    // Type a query that matches ONLY the non-needs-attention item
    const searchInput = screen.getByPlaceholderText('Search title, key, or path…')
    fireEvent.change(searchInput, { target: { value: 'beta' } })

    // Global search must find "Running beta" even though needs-attention is active
    expect(screen.getByText('Running beta')).toBeTruthy()
  })

  it('profile switch clears expanded detail (A→B→A does not resurrect stale cache)', async () => {
    const { fetchInboxRequestDetails } = await import('@/store/inbox')
    const mockFetch = vi.mocked(fetchInboxRequestDetails)

    const detailDefault = {
      coverage: { approval_count: 0, clarification_count: 0, context_anchor: '', errors: [], live_session_count: 0, profile: 'inbox-test-profile', session_key: 'sess-shared' },
      sessions: [{ approvals: [], clarifications: [], live_session_ids: [] }]
    }
    const detailB = {
      coverage: { approval_count: 0, clarification_count: 0, context_anchor: '', errors: [], live_session_count: 0, profile: 'profile-b', session_key: 'sess-shared' },
      sessions: [{ approvals: [], clarifications: [], live_session_ids: [] }]
    }

    mockFetch.mockImplementation(async (_key, profile) => {
      if (profile === 'inbox-test-profile') {return detailDefault}
      if (profile === 'profile-b') {return detailB}
      return null
    })

    const sharedItem = makeItem({ session_key: 'sess-shared', title: 'Shared session' })

    const entryA = makeEntry({
      snapshot: {
        badge: 'none',
        counts: { needs_you: 1, running: 0, waiting: 0, scheduled: 0, total: 1 },
        coverage: { approval_scope: '', clarify_scope: '', connection_scope: '', errors: [], partial: false, profile: 'inbox-test-profile', scanned_sessions: 1 },
        items: [sharedItem]
      }
    })

    const { rerender } = render(
      <MemoryRouter>
        <InboxPanel inbox={entryA} onClose={vi.fn()} />
      </MemoryRouter>
    )

    // Expand under default profile — detail loads
    clickRow('Shared session')
    await screen.findByText('Title')

    // Switch to profile B — cache must be cleared synchronously
    const { $activeGatewayProfile } = await import('@/store/profile')
    vi.mocked($activeGatewayProfile.get).mockReturnValue('profile-b')

    const entryB = makeEntry({
      snapshot: {
        badge: 'none',
        counts: { needs_you: 1, running: 0, waiting: 0, scheduled: 0, total: 1 },
        coverage: { approval_scope: '', clarify_scope: '', connection_scope: '', errors: [], partial: false, profile: 'profile-b', scanned_sessions: 1 },
        items: [sharedItem]
      }
    })

    rerender(
      <MemoryRouter>
        <InboxPanel inbox={entryB} onClose={vi.fn()} />
      </MemoryRouter>
    )

    // Detail panel must be collapsed after scope change — Title label gone
    expect(screen.queryByText('Title')).toBeNull()
  })

  it('connection change clears detail cache even when profile stays the same', async () => {
    const { fetchInboxRequestDetails } = await import('@/store/inbox')
    const mockFetch = vi.mocked(fetchInboxRequestDetails)

    const detailConn1 = {
      coverage: { approval_count: 0, clarification_count: 0, context_anchor: '', errors: [], live_session_count: 0, profile: 'default', session_key: 'sess-x' },
      sessions: [{ approvals: [], clarifications: [], live_session_ids: [] }]
    }

    mockFetch.mockResolvedValue(detailConn1)

    const item = makeItem({ session_key: 'sess-x', title: 'Session X' })

    const entry = makeEntry({
      snapshot: {
        badge: 'none',
        counts: { needs_you: 1, running: 0, waiting: 0, scheduled: 0, total: 1 },
        coverage: { approval_scope: '', clarify_scope: '', connection_scope: '', errors: [], partial: false, profile: 'default', scanned_sessions: 1 },
        items: [item]
      }
    })

    const { rerender } = render(
      <MemoryRouter>
        <InboxPanel inbox={entry} onClose={vi.fn()} />
      </MemoryRouter>
    )

    // Expand session
    clickRow('Session X')
    await screen.findByText('Title')

    // Change connection (same profile)
    const { $connection } = await import('@/store/session')
    vi.mocked($connection.get).mockReturnValue({ connectionId: 'conn-2', baseUrl: '', mode: 'remote' } as never)

    rerender(
      <MemoryRouter>
        <InboxPanel inbox={entry} onClose={vi.fn()} />
      </MemoryRouter>
    )

    // Detail panel must be collapsed after connection change
    expect(screen.queryByText('Title')).toBeNull()
  })

  it('deferred detail response does not populate cache after scope change', async () => {
    const { fetchInboxRequestDetails } = await import('@/store/inbox')
    const mockFetch = vi.mocked(fetchInboxRequestDetails)

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    let resolveFetch: (v: any) => void
    mockFetch.mockImplementation(() => new Promise(resolve => { resolveFetch = resolve }))

    const item = makeItem({ session_key: 'sess-defer', title: 'Deferred session' })

    const entryA = makeEntry({
      snapshot: {
        badge: 'none',
        counts: { needs_you: 1, running: 0, waiting: 0, scheduled: 0, total: 1 },
        coverage: { approval_scope: '', clarify_scope: '', connection_scope: '', errors: [], partial: false, profile: 'profile-a', scanned_sessions: 1 },
        items: [item]
      }
    })

    const { rerender } = render(
      <MemoryRouter>
        <InboxPanel inbox={entryA} onClose={vi.fn()} />
      </MemoryRouter>
    )

    // Expand to trigger loadDetails (fetch is pending)
    clickRow('Deferred session')

    // Switch scope before fetch completes
    const { $activeGatewayProfile } = await import('@/store/profile')
    vi.mocked($activeGatewayProfile.get).mockReturnValue('profile-b')

    const entryB = makeEntry({
      snapshot: {
        badge: 'none',
        counts: { needs_you: 0, running: 0, waiting: 0, scheduled: 0, total: 0 },
        coverage: { approval_scope: '', clarify_scope: '', connection_scope: '', errors: [], partial: false, profile: 'profile-b', scanned_sessions: 0 },
        items: []
      }
    })

    rerender(
      <MemoryRouter>
        <InboxPanel inbox={entryB} onClose={vi.fn()} />
      </MemoryRouter>
    )

    // Now resolve the old deferred fetch
    resolveFetch!({
      inbox: {
        badge: 'none',
        counts: { needs_you: 0, running: 0, waiting: 0, scheduled: 0, total: 0 },
        coverage: { approval_scope: '', clarify_scope: '', connection_scope: '', errors: [], partial: false, profile: 'profile-a', scanned_sessions: 1 },
        items: []
      }
    })

    // The old response should NOT populate the cache — detail panel stays collapsed
    expect(screen.queryByText('Title')).toBeNull()
  })

  // ── Defect 1: Gateway object reference scope ───────────────────────────────

  it('gateway object reference change clears details even with same connectionId/profile', async () => {
    const { fetchInboxRequestDetails } = await import('@/store/inbox')
    const mockFetch = vi.mocked(fetchInboxRequestDetails)

    const detail = {
      coverage: { approval_count: 0, clarification_count: 0, context_anchor: '', errors: [], live_session_count: 0, profile: 'default', session_key: 'sess-gw' },
      sessions: [{ approvals: [], clarifications: [], live_session_ids: [] }]
    }
    mockFetch.mockResolvedValue(detail)

    const { $gateway } = await import('@/store/gateway')
    const gwA = { request: vi.fn(), connectionId: 'conn-1', baseUrl: '' }
    const gwB = { request: vi.fn(), connectionId: 'conn-1', baseUrl: '' } // same IDs, different object
    $gateway.set(gwA as never)

    const item = makeItem({ session_key: 'sess-gw', title: 'GW session' })
    const entry = makeEntry({
      snapshot: {
        badge: 'none',
        counts: { needs_you: 1, running: 0, waiting: 0, scheduled: 0, total: 1 },
        coverage: { approval_scope: '', clarify_scope: '', connection_scope: '', errors: [], partial: false, profile: 'default', scanned_sessions: 1 },
        items: [item]
      }
    })

    render(
      <MemoryRouter>
        <InboxPanel inbox={entry} onClose={vi.fn()} />
      </MemoryRouter>
    )

    // Expand — detail loads
    clickRow('GW session')
    await screen.findByText('Title')

    // Replace gateway object (simulates reconnect with same configured IDs)
    await act(async () => {
      $gateway.set(gwB as never)
    })

    // Detail must be collapsed — gateway identity changed
    await waitFor(() => {
      expect(screen.queryByText('Title')).toBeNull()
    })
  })

  // ── Defect 2: Generation invalidation A→B→A with deferred result ──────────

  it('deferred A result rejected after A→B→A scope cycle (generation invalidation)', async () => {
    const { fetchInboxRequestDetails } = await import('@/store/inbox')
    const mockFetch = vi.mocked(fetchInboxRequestDetails)

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const fetchResolvers: Array<(v: any) => void> = []
    let callCount = 0
    mockFetch.mockImplementation(() => {
      callCount++
      return new Promise(resolve => { fetchResolvers.push(resolve) })
    })

    const itemA = makeItem({ session_key: 'sess-cycle', title: 'Cycle session' })

    const entryA = makeEntry({
      snapshot: {
        badge: 'none',
        counts: { needs_you: 1, running: 0, waiting: 0, scheduled: 0, total: 1 },
        coverage: { approval_scope: '', clarify_scope: '', connection_scope: '', errors: [], partial: false, profile: 'profile-a', scanned_sessions: 1 },
        items: [itemA]
      }
    })

    const { rerender } = render(
      <MemoryRouter>
        <InboxPanel inbox={entryA} onClose={vi.fn()} />
      </MemoryRouter>
    )

    // Expand under profile A — first fetch starts (pending)
    clickRow('Cycle session')

    // Switch to profile B — scope changes, generation increments
    const { $activeGatewayProfile } = await import('@/store/profile')
    vi.mocked($activeGatewayProfile.get).mockReturnValue('profile-b')

    const entryB = makeEntry({
      snapshot: {
        badge: 'none',
        counts: { needs_you: 0, running: 0, waiting: 0, scheduled: 0, total: 0 },
        coverage: { approval_scope: '', clarify_scope: '', connection_scope: '', errors: [], partial: false, profile: 'profile-b', scanned_sessions: 0 },
        items: []
      }
    })

    rerender(
      <MemoryRouter>
        <InboxPanel inbox={entryB} onClose={vi.fn()} />
      </MemoryRouter>
    )

    // Switch BACK to profile A — scope changes again, generation increments again
    vi.mocked($activeGatewayProfile.get).mockReturnValue('profile-a')

    rerender(
      <MemoryRouter>
        <InboxPanel inbox={entryA} onClose={vi.fn()} />
      </MemoryRouter>
    )

    // Re-expand the same session — second fetch starts (pending)
    clickRow('Cycle session')
    expect(callCount).toBe(2)

    // Resolve the SECOND fetch (current scope) with valid data
    fetchResolvers[1]!({
      coverage: { approval_count: 0, clarification_count: 0, context_anchor: '', errors: [], live_session_count: 0, profile: 'profile-a', session_key: 'sess-cycle' },
      sessions: [{ approvals: [], clarifications: [], live_session_ids: [] }]
    })

    await vi.waitFor(() => {
      expect(screen.getByText('Title')).toBeTruthy()
    })

    // Now resolve the FIRST (stale, seq=1) fetch with DIFFERENT data
    fetchResolvers[0]!({
      coverage: { approval_count: 99, clarification_count: 99, context_anchor: 'stale', errors: [], live_session_count: 0, profile: 'profile-a', session_key: 'sess-cycle' },
      sessions: [{ approvals: [], clarifications: [], live_session_ids: [] }]
    })

    // Wait a tick for the stale resolution to (not) apply
    await new Promise(r => setTimeout(r, 10))

    // The stale data must NOT have overwritten the current data.
    // "Title" should still be present (from the valid second fetch), not replaced by stale data.
    // The key assertion: context_anchor from stale data ('stale') must NOT appear.
    expect(screen.queryByText('stale')).toBeNull()
  })

  // ── Defect 3: handleNeedsAttentionClick idempotent ─────────────────────────

  it('clicking Needs attention when already selected stays selected (idempotent)', () => {
    const needsItem = makeItem({ lanes: ['needs_you'], session_key: 'sess-n1', title: 'Needs only' })
    const runningItem = makeItem({ lanes: ['running'], session_key: 'sess-r1', title: 'Running only' })

    const entry = makeEntry({
      snapshot: {
        badge: 'amber',
        counts: { needs_you: 1, running: 1, waiting: 0, scheduled: 0, total: 2 },
        coverage: { approval_scope: '', clarify_scope: '', connection_scope: '', errors: [], partial: false, profile: 'default', scanned_sessions: 2 },
        items: [needsItem, runningItem]
      }
    })

    renderPanel(entry)

    // Activate needs attention
    fireEvent.click(screen.getByText('Needs attention'))
    expect(screen.getByText('Needs only')).toBeTruthy()
    expect(screen.queryByText('Running only')).toBeNull()

    // Click again — must STAY in needs-attention mode (idempotent, not toggle)
    fireEvent.click(screen.getByText('Needs attention'))
    expect(screen.getByText('Needs only')).toBeTruthy()
    expect(screen.queryByText('Running only')).toBeNull()
  })
})
