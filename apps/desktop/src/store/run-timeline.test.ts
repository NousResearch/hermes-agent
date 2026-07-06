import { describe, expect, it } from 'vitest'

import { buildHistoricalChildLanes, buildRunTimeline, type HistoricalChildInput, type ParentToolInput, toolFamily } from './run-timeline'
import type { SubagentProgress } from './subagents'

const CREATED_MS = 1_700_000_000_000 // fixed backend epoch ms

function sub(overrides: Partial<SubagentProgress> & { id: string }): SubagentProgress {
  return {
    parentId: null,
    goal: overrides.id,
    status: 'running',
    taskCount: 1,
    taskIndex: 0,
    startedAt: 0,
    updatedAt: 0,
    filesRead: [],
    filesWritten: [],
    stream: [],
    ...overrides
  }
}

describe('toolFamily', () => {
  it('maps known tools and prefixes to families', () => {
    expect(toolFamily('terminal')).toBe('terminal')
    expect(toolFamily('read_file')).toBe('file')
    expect(toolFamily('browser_navigate')).toBe('browser')
    expect(toolFamily('browser_anything_new')).toBe('browser')
    expect(toolFamily('web_search')).toBe('web')
    expect(toolFamily('delegate_task')).toBe('delegation')
    expect(toolFamily('some_unknown_tool')).toBe('other')
  })
})

describe('buildRunTimeline', () => {
  const parentTools: ParentToolInput[] = [
    { toolCallId: 't1', name: 'terminal', startedAt: 1_700_000_000, durationS: 2 },
    { toolCallId: 't2', name: 'read_file', startedAt: 1_700_000_005, durationS: 1 },
    { toolCallId: 't3', name: 'browser_navigate', startedAt: 1_700_000_010, durationS: 30 } // outlier
  ]

  it('parent lane renders per-tool blocks; child lanes render bars', () => {
    const subs: SubagentProgress[] = [
      sub({ id: 'c1', parentId: null, goal: 'lens A', status: 'completed', durationSeconds: 12, toolCount: 9 })
    ]

    const tl = buildRunTimeline(subs, parentTools, CREATED_MS, 'parent', 'Parent session')

    const parent = tl.lanes.find(l => l.id === 'parent')!
    expect(parent.kind).toBe('blocks')
    expect(parent.blocks).toHaveLength(3)
    expect(parent.bar).toBeNull()

    const child = tl.lanes.find(l => l.id === 'c1')!
    expect(child.kind).toBe('bar')
    expect(child.blocks).toHaveLength(0)
    expect(child.bar).toEqual({ toolCount: 9, isOutlier: false })
  })

  it('uses backend epoch for block startMs (single clock, no Date.now)', () => {
    const tl = buildRunTimeline([], parentTools, CREATED_MS, 'parent', 'Parent')
    const parent = tl.lanes[0]
    // started_at 1_700_000_000 s -> 1_700_000_000_000 ms
    expect(parent.blocks[0].startMs).toBe(1_700_000_000_000)
    expect(parent.blocks[0].durationMs).toBe(2000)
  })

  it('flags a slow parent tool as an outlier at the default 20s threshold', () => {
    const tl = buildRunTimeline([], parentTools, CREATED_MS, 'parent', 'Parent')
    const parent = tl.lanes[0]
    expect(parent.blocks[2].isOutlier).toBe(true) // 30s > 20s
    expect(parent.blocks[0].isOutlier).toBe(false) // 2s
  })

  it('flags a slow child lane bar as an outlier', () => {
    const subs = [sub({ id: 'slow', durationSeconds: 40, toolCount: 3, status: 'completed' })]
    const tl = buildRunTimeline(subs, [], CREATED_MS, 'parent', 'Parent')
    const child = tl.lanes.find(l => l.id === 'slow')!
    expect(child.bar?.isOutlier).toBe(true)
  })

  it('nests grandchildren with increasing depth', () => {
    const subs: SubagentProgress[] = [
      sub({ id: 'c1', parentId: null, goal: 'child', status: 'completed', durationSeconds: 5, toolCount: 2 }),
      sub({ id: 'g1', parentId: 'c1', goal: 'grandchild', status: 'completed', durationSeconds: 2, toolCount: 1 })
    ]

    const tl = buildRunTimeline(subs, [], CREATED_MS, 'parent', 'Parent')
    const parent = tl.lanes.find(l => l.id === 'parent')!
    const c1 = tl.lanes.find(l => l.id === 'c1')!
    const g1 = tl.lanes.find(l => l.id === 'g1')!

    expect(parent.depth).toBe(0)
    expect(c1.depth).toBe(1)
    expect(g1.depth).toBe(2)
    expect(g1.parentLaneId).toBe('c1')
  })

  it('marks a running child lane with null endMs', () => {
    const subs = [sub({ id: 'r1', status: 'running', toolCount: 1 })]
    const tl = buildRunTimeline(subs, [], CREATED_MS, 'parent', 'Parent')
    const child = tl.lanes.find(l => l.id === 'r1')!
    expect(child.running).toBe(true)
    expect(child.endMs).toBeNull()
  })

  it('running parent tool (no durationS) has null durationMs and lays out sequentially with no started_at', () => {
    const noTiming: ParentToolInput[] = [
      { toolCallId: 'a', name: 'terminal' }, // no startedAt, no durationS
      { toolCallId: 'b', name: 'read_file', durationS: 3 }
    ]

    const tl = buildRunTimeline([], noTiming, CREATED_MS, 'parent', 'Parent')
    const parent = tl.lanes[0]
    // pre-P1 fallback: first block anchored at session start
    expect(parent.blocks[0].startMs).toBe(CREATED_MS)
    expect(parent.blocks[0].durationMs).toBeNull()
    // second block follows the first (sequential fallback)
    expect(parent.blocks[1].startMs).toBe(CREATED_MS)
  })

  it('live append: adding a new subagent produces a new lane on rebuild', () => {
    const before = buildRunTimeline([], parentTools, CREATED_MS, 'parent', 'Parent')
    expect(before.lanes).toHaveLength(1)

    const after = buildRunTimeline(
      [sub({ id: 'late', status: 'running', toolCount: 0 })],
      parentTools,
      CREATED_MS,
      'parent',
      'Parent'
    )

    expect(after.lanes).toHaveLength(2)
    expect(after.lanes.some(l => l.id === 'late')).toBe(true)
  })
})

describe('buildHistoricalChildLanes', () => {
  const S = 1_700_000_000 // backend epoch seconds

  const child = (overrides: Partial<HistoricalChildInput> & { id: string }): HistoricalChildInput => ({
    label: overrides.id,
    startedAtS: S,
    endedAtS: S + 60,
    toolCount: 5,
    ...overrides
  })

  it('maps a finished child to a bar with real start/end (seconds -> ms) and toolCount', () => {
    const [lane] = buildHistoricalChildLanes([child({ id: 'c1', startedAtS: S, endedAtS: S + 42, toolCount: 22 })], 'parent', null)

    expect(lane.kind).toBe('bar')
    expect(lane.parentLaneId).toBe('parent')
    expect(lane.depth).toBe(1)
    expect(lane.startMs).toBe(S * 1000)
    expect(lane.endMs).toBe((S + 42) * 1000)
    expect(lane.running).toBe(false)
    expect(lane.bar?.toolCount).toBe(22)
    expect(lane.blocks).toEqual([])
  })

  it('flags a long child as an outlier at the default 20s threshold', () => {
    const [lane] = buildHistoricalChildLanes([child({ id: 'slow', startedAtS: S, endedAtS: S + 21 })], 'parent', null)

    expect(lane.bar?.isOutlier).toBe(true)
  })

  it('does not flag a short child as an outlier', () => {
    const [lane] = buildHistoricalChildLanes([child({ id: 'fast', startedAtS: S, endedAtS: S + 3 })], 'parent', null)

    expect(lane.bar?.isOutlier).toBe(false)
  })

  it('renders a running child (endedAtS null) as a running bar with endMs null under a live parent', () => {
    const [lane] = buildHistoricalChildLanes([child({ id: 'live', endedAtS: null })], 'parent', null)

    expect(lane.running).toBe(true)
    expect(lane.endMs).toBeNull()
  })

  it('clamps a running child under a FINISHED parent to the parent end (no stretch to now)', () => {
    const parentEndMs = (S + 100) * 1000
    const [lane] = buildHistoricalChildLanes([child({ id: 'orphan', startedAtS: S, endedAtS: null })], 'parent', parentEndMs)

    // Still marked running (no ended_at), but the bar is bounded by the parent's end.
    expect(lane.running).toBe(true)
    expect(lane.endMs).toBe(parentEndMs)
  })

  it('returns no lanes for an empty child list', () => {
    expect(buildHistoricalChildLanes([], 'parent', null)).toEqual([])
  })
})
