// @vitest-environment jsdom
import { describe, expect, it, vi } from 'vitest'

import type { ProfileInfo, SessionInfo } from '@/types/hermes'
import type { KanbanBoardPayload, KanbanCard } from '@/types/kanban'

// Keep store/profile's side-effecting imports inert (same pattern as
// store/profile.test.ts): the gateway socket layer and the REST query client
// must not run for real in a unit test.
vi.mock('@/store/gateway', () => ({
  $gateway: { get: () => null },
  ensureGatewayForProfile: vi.fn(async () => undefined)
}))
vi.mock('@/hermes', () => ({
  getProfiles: vi.fn(async () => ({ profiles: [] })),
  setApiRequestProfile: vi.fn()
}))
vi.mock('@/lib/query-client', () => ({ queryClient: { invalidateQueries: vi.fn() } }))
vi.mock('@/store/starmap', () => ({ resetStarmapGraph: vi.fn() }))

const { buildProfileOverviewRows, summarizeKanbanBoard } = await import('./overview-data')
const { KANBAN_COLUMNS } = await import('@/store/kanban')

function session(id: string, profile: string | undefined, extra: Partial<SessionInfo> = {}): SessionInfo {
  return {
    ended_at: null,
    id,
    input_tokens: 0,
    is_active: true,
    last_active: 100,
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

function profile(name: string, isDefault = false): ProfileInfo {
  return {
    has_env: false,
    is_default: isDefault,
    model: null,
    name,
    path: `/tmp/hermes/${name}`,
    provider: null,
    skill_count: 0
  }
}

function card(id: string, status: string, extra: Partial<KanbanCard> = {}): KanbanCard {
  return {
    id,
    title: `Task ${id}`,
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

function board(tasksByColumn: Record<string, KanbanCard[]>): KanbanBoardPayload {
  return {
    columns: KANBAN_COLUMNS.map(name => ({ name, tasks: tasksByColumn[name] ?? [] })),
    tenants: [],
    assignees: [],
    latest_event_id: 1,
    now: 1000
  }
}

describe('buildProfileOverviewRows', () => {
  it('aggregates working/attention/last-activity per profile in rail order', () => {
    const rows = buildProfileOverviewRows({
      sessions: [
        session('s1', undefined, { last_active: 300 }),
        session('s2', 'default', { last_active: 100 }),
        session('s3', 'coder', { last_active: 900 }),
        session('s4', 'coder', { last_active: 500 }),
        session('s5', 'writer', { last_active: 0, started_at: 40 })
      ],
      profiles: [profile('default', true), profile('writer'), profile('coder'), profile('idle-one')],
      profileOrder: ['coder', 'writer'],
      workingIds: ['s3', 's4', 's1'],
      attentionIds: ['s4', 's5'],
      profileTotals: {}
    })

    expect(rows.map(row => row.key)).toEqual(['default', 'coder', 'writer', 'idle-one'])

    const [def, coder, writer, idle] = rows

    expect(def).toMatchObject({ isDefault: true, workingCount: 1, attentionCount: 0, sessionCount: 2 })
    expect(def.lastActive).toBe(300)

    // s4 sits in BOTH sets (turn running while blocked on input): needs-input
    // wins, same precedence as deriveProfileActivity / the rail badge — so it
    // counts once, under attention, never inflating working.
    expect(coder).toMatchObject({ isDefault: false, workingCount: 1, attentionCount: 1, sessionCount: 2 })
    expect(coder.lastActive).toBe(900)

    // last_active 0 falls back to started_at.
    expect(writer).toMatchObject({ workingCount: 0, attentionCount: 1, sessionCount: 1, lastActive: 40 })

    // A profile with no sessions still gets a (zeroed) row.
    expect(idle).toMatchObject({ workingCount: 0, attentionCount: 0, sessionCount: 0, lastActive: null })
  })

  it('prefers aggregator profile totals over the loaded count', () => {
    const rows = buildProfileOverviewRows({
      sessions: [session('s1', 'coder')],
      profiles: [profile('default', true), profile('coder')],
      profileOrder: [],
      workingIds: [],
      attentionIds: [],
      profileTotals: { coder: 42, default: 7 }
    })

    expect(rows.find(row => row.key === 'coder')?.sessionCount).toBe(42)
    expect(rows.find(row => row.key === 'default')?.sessionCount).toBe(7)
  })

  it('appends session-only profiles the profile list does not know yet', () => {
    const rows = buildProfileOverviewRows({
      sessions: [session('s1', 'zeta'), session('s2', 'alpha')],
      profiles: [profile('default', true)],
      profileOrder: [],
      workingIds: ['s1'],
      attentionIds: [],
      profileTotals: {}
    })

    expect(rows.map(row => row.key)).toEqual(['default', 'alpha', 'zeta'])
    expect(rows.find(row => row.key === 'zeta')?.workingCount).toBe(1)
  })

  it('always yields a default row even with an empty profile list', () => {
    const rows = buildProfileOverviewRows({
      sessions: [],
      profiles: [],
      profileOrder: [],
      workingIds: [],
      attentionIds: [],
      profileTotals: {}
    })

    expect(rows).toEqual([
      {
        key: 'default',
        isDefault: true,
        workingCount: 0,
        attentionCount: 0,
        sessionCount: 0,
        lastActive: null
      }
    ])
  })
})

describe('summarizeKanbanBoard', () => {
  it('zero-fills per-status counts in canonical column order', () => {
    const summary = summarizeKanbanBoard(
      board({ todo: [card('a', 'todo'), card('b', 'todo')], done: [card('c', 'done')] })
    )

    expect(summary.counts.map(entry => entry.name)).toEqual([...KANBAN_COLUMNS])
    expect(summary.counts.find(entry => entry.name === 'todo')?.count).toBe(2)
    expect(summary.counts.find(entry => entry.name === 'triage')?.count).toBe(0)
    expect(summary.total).toBe(3)
  })

  it('ranks hot cards review-first, then priority desc, then oldest, capped', () => {
    const summary = summarizeKanbanBoard(
      board({
        review: [
          card('r_low', 'review', { priority: 0, created_at: 5 }),
          card('r_high', 'review', { priority: 2, created_at: 9 })
        ],
        blocked: [
          card('b_old', 'blocked', { priority: 1, created_at: 1 }),
          card('b_new', 'blocked', { priority: 1, created_at: 8 }),
          card('b_zero', 'blocked', { priority: 0, created_at: 2 })
        ]
      }),
      4
    )

    expect(summary.hotCards.map(entry => entry.id)).toEqual(['r_high', 'r_low', 'b_old', 'b_new'])
  })

  it('returns no hot cards for an empty board', () => {
    expect(summarizeKanbanBoard(board({})).hotCards).toEqual([])
  })
})
