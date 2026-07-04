// @vitest-environment jsdom
import { cleanup, render, screen, waitFor } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ProfileInfo, SessionInfo } from '@/types/hermes'
import type { KanbanBoardPayload, KanbanCard } from '@/types/kanban'

// Keep store/profile's side-effecting imports inert: the gateway socket layer
// and the REST query client must not run for real (same idiom as
// store/profile.test.ts).
const ensureGatewayForProfile = vi.fn(async () => undefined)
const $gateway = atom<unknown>({ id: 'live-socket' })

vi.mock('@/store/gateway', () => ({ $gateway, ensureGatewayForProfile }))
vi.mock('@/hermes', () => ({
  getProfiles: vi.fn(async () => ({
    profiles: [
      {
        has_env: false,
        is_default: true,
        model: null,
        name: 'default',
        path: '/tmp/h/default',
        provider: null,
        skill_count: 0
      },
      {
        has_env: false,
        is_default: false,
        model: null,
        name: 'coder',
        path: '/tmp/h/coder',
        provider: null,
        skill_count: 0
      }
    ]
  })),
  setApiRequestProfile: vi.fn()
}))
vi.mock('@/lib/query-client', () => ({ queryClient: { invalidateQueries: vi.fn() } }))
vi.mock('@/store/starmap', () => ({ resetStarmapGraph: vi.fn() }))

const { OverviewPanel } = await import('./overview')

const {
  $kanbanActiveBoard,
  $kanbanBoard,
  $kanbanBoardError,
  $kanbanRefreshing,
  $kanbanSelectedTaskId,
  KANBAN_COLUMNS
} = await import('@/store/kanban')

const { $activeGatewayProfile, $newChatProfile, $profileOrder, $profiles, $showAllProfiles } =
  await import('@/store/profile')

const { $attentionSessionIds, $sessionProfileTotals, $sessions, $workingSessionIds } = await import('@/store/session')

function session(id: string, profile: string | undefined, extra: Partial<SessionInfo> = {}): SessionInfo {
  return {
    ended_at: null,
    id,
    input_tokens: 0,
    is_active: true,
    last_active: Math.floor(Date.now() / 1000),
    message_count: 1,
    model: null,
    output_tokens: 0,
    preview: null,
    profile,
    source: null,
    started_at: 50,
    title: null,
    tool_call_count: 0,
    ...extra
  }
}

function card(id: string, title: string, status: string, extra: Partial<KanbanCard> = {}): KanbanCard {
  return {
    id,
    title,
    body: null,
    assignee: null,
    status,
    priority: 0,
    created_by: null,
    created_at: 1,
    started_at: null,
    completed_at: null,
    workspace_kind: 'scratch',
    workspace_path: null,
    claim_lock: null,
    claim_expires: null,
    tenant: null,
    ...extra
  }
}

const boardPayload: KanbanBoardPayload = {
  columns: KANBAN_COLUMNS.map(name => ({
    name,
    tasks:
      name === 'review'
        ? [card('t_rev', 'Ship the login fix', 'review')]
        : name === 'todo'
          ? [card('t_todo', 'Backlog item', 'todo')]
          : []
  })),
  tenants: [],
  assignees: [],
  latest_event_id: 7,
  now: 1000
}

const profiles: ProfileInfo[] = [
  {
    has_env: false,
    is_default: true,
    model: null,
    name: 'default',
    path: '/tmp/h/default',
    provider: null,
    skill_count: 0
  },
  {
    has_env: false,
    is_default: false,
    model: null,
    name: 'coder',
    path: '/tmp/h/coder',
    provider: null,
    skill_count: 0
  }
]

describe('OverviewPanel', () => {
  let api: ReturnType<typeof vi.fn>
  let boardFails: boolean

  beforeEach(() => {
    boardFails = false
    api = vi.fn(async (request: { path: string }) => {
      if (request.path.startsWith('/api/plugins/kanban/boards')) {
        if (boardFails) {
          throw new Error('404: {"detail":"Not Found"}')
        }

        return {
          boards: [{ slug: 'default', name: 'Default', description: '', icon: '', color: '', is_current: true }]
        }
      }

      if (request.path.startsWith('/api/plugins/kanban/board')) {
        if (boardFails) {
          throw new Error('404: {"detail":"Not Found"}')
        }

        return boardPayload
      }

      throw new Error(`unexpected path ${request.path}`)
    })
    Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { api } })
    window.localStorage.clear()

    $sessions.set([
      session('s_def', 'default'),
      session('s_work', 'coder'),
      session('s_ask', 'coder', { last_active: Math.floor(Date.now() / 1000) - 60 })
    ])
    $workingSessionIds.set(['s_work'])
    $attentionSessionIds.set(['s_ask'])
    $sessionProfileTotals.set({ coder: 12, default: 3 })
    $profiles.set(profiles)
    $profileOrder.set([])
    $activeGatewayProfile.set('default')
    $showAllProfiles.set(false)
    $newChatProfile.set(null)

    $kanbanActiveBoard.set(null)
    $kanbanBoard.set(null)
    $kanbanBoardError.set(null)
    $kanbanRefreshing.set(false)
    $kanbanSelectedTaskId.set(null)

    ensureGatewayForProfile.mockClear()
  })

  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
    Reflect.deleteProperty(window, 'hermesDesktop')
  })

  it('renders one activity row per profile with working/attention badges', async () => {
    render(<OverviewPanel onClose={() => undefined} />)

    expect(await screen.findByText('default')).toBeTruthy()
    expect(screen.getByText('coder')).toBeTruthy()

    // coder: one working, one needing input; default: idle.
    expect(screen.getByText('1 working')).toBeTruthy()
    expect(screen.getByText('1 needs input')).toBeTruthy()
    expect(screen.getByText('Idle')).toBeTruthy()

    // Aggregator totals win over the loaded count.
    expect(screen.getByText(/12 sessions/)).toBeTruthy()
    expect(screen.getByText(/3 sessions/)).toBeTruthy()

    // The active profile is badged and has no switch button.
    expect(screen.getByText('Current')).toBeTruthy()
    expect(screen.getAllByRole('button', { name: 'Switch to coder' })).toHaveLength(1)
    expect(screen.queryByRole('button', { name: 'Switch to default' })).toBeNull()
  })

  it('quick-switches profiles via selectProfile and closes the panel', async () => {
    const onClose = vi.fn()

    render(<OverviewPanel onClose={onClose} />)

    const switchButton = await screen.findByRole('button', { name: 'Switch to coder' })

    switchButton.click()

    expect(onClose).toHaveBeenCalledTimes(1)
    expect($newChatProfile.get()).toBe('coder')
    await waitFor(() => expect(ensureGatewayForProfile).toHaveBeenCalledWith('coder'))
  })

  it('shows the kanban summary strip and opens a hot card on the board', async () => {
    const onNavigateRoute = vi.fn()

    render(<OverviewPanel onClose={() => undefined} onNavigateRoute={onNavigateRoute} />)

    const chip = await screen.findByRole('button', { name: 'Open "Ship the login fix" on the kanban board' })

    // Column counts from the mocked payload (1 todo; 'Review' also appears on
    // the chip's status label, hence getAllByText).
    expect(screen.getByText('Todo')).toBeTruthy()
    expect(screen.getAllByText('Review').length).toBeGreaterThanOrEqual(1)

    chip.click()

    expect($kanbanSelectedTaskId.get()).toBe('t_rev')
    expect(onNavigateRoute).toHaveBeenCalledWith('/kanban')
  })

  it('falls back gracefully when the kanban API is unavailable', async () => {
    boardFails = true

    render(<OverviewPanel onClose={() => undefined} />)

    expect(await screen.findByText('Kanban is not available on this backend.')).toBeTruthy()

    // Profile rows still render — the overview degrades, it does not error out.
    expect(screen.getByText('coder')).toBeTruthy()
    expect(screen.queryByText('Open board')).toBeNull()
  })
})
