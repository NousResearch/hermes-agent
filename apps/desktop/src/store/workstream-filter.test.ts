import { beforeEach, describe, expect, it } from 'vitest'

import type { SessionInfo } from '@/types/hermes'

import { WORKSTREAM_STATE_META, type WorkstreamActivity, type WorkstreamState } from './workstream'
import {
  $workstreamFilter,
  $workstreamVisibleSessionIds,
  adjacentWorkstreamSessionId,
  collectRenderedWorkstreamSessionIds,
  collectWorkstreamVisibleSessionIds,
  cycleWorkstreamFilter,
  displaySessionsForWorkstreamFilter,
  filterSessionsByWorkstream,
  setWorkstreamVisibleSessionIds,
  WORKSTREAM_FILTERS,
  workstreamFilterPredicate,
  type WorkstreamFilterRuntime
} from './workstream-filter'
import { $workstreamMetadata } from './workstream-metadata'

const session = (id: string): SessionInfo =>
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
    title: id,
    tool_call_count: 0
  }) as SessionInfo

const activity = (state: WorkstreamState): WorkstreamActivity => ({
  activeSubagentCount: 0,
  activeTodoCount: 0,
  completedTodoCount: 0,
  failedSubagentCount: 0,
  icon: WORKSTREAM_STATE_META[state].icon,
  isWorking: false,
  label: WORKSTREAM_STATE_META[state].label,
  needsInput: false,
  sessionId: state,
  state,
  tone: WORKSTREAM_STATE_META[state].tone,
  totalSubagentCount: 0,
  totalTodoCount: 0
})

const runtime = (overrides: Partial<WorkstreamFilterRuntime> = {}): WorkstreamFilterRuntime => ({
  activeSessionId: null,
  attentionSessionIds: [],
  metadataBySession: {},
  selectedStoredSessionId: null,
  subagentsBySession: {},
  todosBySession: {},
  workingSessionIds: [],
  ...overrides
})

describe('workstream filters', () => {
  beforeEach(() => {
    $workstreamFilter.set('all')
    $workstreamMetadata.set({})
    $workstreamVisibleSessionIds.set([])
  })

  it('cycles through the shipped filter order', () => {
    const seen = [cycleWorkstreamFilter()]

    for (let index = 1; index < WORKSTREAM_FILTERS.length; index += 1) {
      seen.push(cycleWorkstreamFilter())
    }

    expect(seen).toEqual(['active', 'blocked', 'review', 'closed', 'safe-delete', 'all'])
    expect($workstreamFilter.get()).toBe('all')
  })

  it('keeps active work distinct from blocked and review states', () => {
    expect(workstreamFilterPredicate('active', activity('work'), 'active')).toBe(true)
    expect(workstreamFilterPredicate('active', activity('delegate'), 'active')).toBe(true)
    expect(workstreamFilterPredicate('active', activity('done'), 'active')).toBe(true)
    expect(workstreamFilterPredicate('active', activity('blocked'), 'active')).toBe(false)
    expect(workstreamFilterPredicate('active', activity('warn'), 'active')).toBe(false)
    expect(workstreamFilterPredicate('active', activity('verify'), 'active')).toBe(false)
    expect(workstreamFilterPredicate('active', activity('plan_review'), 'active')).toBe(false)
    expect(workstreamFilterPredicate('active', activity('work'), 'closed')).toBe(false)
  })

  it('filters blocked and review buckets by actionable state', () => {
    expect(workstreamFilterPredicate('blocked', activity('blocked'), 'active')).toBe(true)
    expect(workstreamFilterPredicate('blocked', activity('warn'), 'active')).toBe(true)
    expect(workstreamFilterPredicate('blocked', activity('restart'), 'restart_required')).toBe(true)
    expect(workstreamFilterPredicate('blocked', activity('work'), 'active')).toBe(false)
    expect(workstreamFilterPredicate('review', activity('verify'), 'active')).toBe(true)
    expect(workstreamFilterPredicate('review', activity('plan_review'), 'active')).toBe(true)
    expect(workstreamFilterPredicate('review', activity('blocked'), 'active')).toBe(false)
  })

  it('separates closed and safe-delete metadata despite the shared close display state', () => {
    const close = activity('close')

    expect(workstreamFilterPredicate('closed', close, 'closed')).toBe(true)
    expect(workstreamFilterPredicate('safe-delete', close, 'closed')).toBe(false)
    expect(workstreamFilterPredicate('closed', close, 'safe_delete')).toBe(false)
    expect(workstreamFilterPredicate('safe-delete', close, 'safe_delete')).toBe(true)
  })

  it('uses the selected stored-session runtime id when filtering live activity', () => {
    const sessions = [session('stored-1'), session('other')]

    const filtered = filterSessionsByWorkstream(
      sessions,
      'blocked',
      runtime({
        activeSessionId: 'runtime-1',
        attentionSessionIds: ['runtime-1'],
        selectedStoredSessionId: 'stored-1'
      })
    )

    expect(filtered.map(item => item.id)).toEqual(['stored-1'])
  })

  it('tracks adjacent visible session ids for workstream navigation', () => {
    setWorkstreamVisibleSessionIds(['a', 'b', 'c'])

    expect(adjacentWorkstreamSessionId('b', 1)).toBe('c')
    expect(adjacentWorkstreamSessionId('b', -1)).toBe('a')
    expect(adjacentWorkstreamSessionId('c', 1)).toBe('a')
    expect(adjacentWorkstreamSessionId(null, 1)).toBe('a')
  })

  it('does not cap messaging sections before applying an active workstream filter', () => {
    const sessions = [session('hidden-match'), session('visible-a'), session('visible-b')]

    expect(displaySessionsForWorkstreamFilter(sessions, 2, 'all').map(item => item.id)).toEqual([
      'hidden-match',
      'visible-a'
    ])
    expect(displaySessionsForWorkstreamFilter(sessions, 2, 'blocked').map(item => item.id)).toEqual([
      'hidden-match',
      'visible-a',
      'visible-b'
    ])
  })

  it('collects visible navigation ids from every rendered session section in order', () => {
    const ids = collectWorkstreamVisibleSessionIds(
      [[session('pinned'), session('dupe')], [session('dupe'), session('recent')], [session('messaging')]],
      'all',
      runtime()
    )

    expect(ids).toEqual(['pinned', 'dupe', 'recent', 'messaging'])
  })

  it('collects navigation ids from rendered session rows only', () => {
    const root = document.createElement('div')
    root.innerHTML = `
      <div data-workstream-session-id="visible-a"></div>
      <div data-workstream-session-id="visible-b"></div>
      <div data-workstream-session-id="visible-a"></div>
    `

    expect(collectRenderedWorkstreamSessionIds(root)).toEqual(['visible-a', 'visible-b'])
  })
})
