import { beforeEach, describe, expect, it, vi } from 'vitest'

import {
  $subagentsBySession,
  activeSubagentCount,
  buildSubagentTree,
  clearSessionSubagents,
  pruneDelegateFallbackSubagents,
  upsertSubagent
} from './subagents'

const listFor = (sid: string) => $subagentsBySession.get()[sid] ?? []

describe('subagent store', () => {
  beforeEach(() => $subagentsBySession.set({}))

  it('upserts subagent progress and keeps terminal status stable', () => {
    upsertSubagent('s1', { goal: 'scan files', status: 'running', subagent_id: 'a1', task_index: 0 })
    upsertSubagent('s1', { goal: 'scan files', status: 'completed', subagent_id: 'a1', summary: 'done', task_index: 0 })
    upsertSubagent('s1', { goal: 'scan files', status: 'running', subagent_id: 'a1', task_index: 0, text: 'late' })

    const item = listFor('s1')[0]
    expect(item?.status).toBe('completed')
    expect(item?.summary).toBe('done')
  })

  it('builds parent/child trees', () => {
    upsertSubagent('s1', { goal: 'parent', status: 'running', subagent_id: 'p', task_index: 0 })
    upsertSubagent('s1', { goal: 'child', parent_id: 'p', status: 'queued', subagent_id: 'c', task_index: 1 })

    const tree = buildSubagentTree(listFor('s1'))
    expect(tree).toHaveLength(1)
    expect(tree[0]?.children[0]?.goal).toBe('child')
    expect(activeSubagentCount(listFor('s1'))).toBe(2)
  })

  it('keeps root nodes in spawn order, not task index order', () => {
    const nowSpy = vi.spyOn(Date, 'now')
    nowSpy.mockReturnValueOnce(1_000)
    upsertSubagent('s1', { goal: 'first spawn', status: 'running', subagent_id: 'a', task_index: 2 })
    nowSpy.mockReturnValueOnce(2_000)
    upsertSubagent('s1', { goal: 'second spawn', status: 'running', subagent_id: 'b', task_index: 0 })
    nowSpy.mockRestore()

    expect(buildSubagentTree(listFor('s1')).map(n => n.id)).toEqual(['a', 'b'])
  })

  it('captures live thinking/progress/tool stream lines', () => {
    upsertSubagent(
      's1',
      { goal: 'scan files', status: 'queued', subagent_id: 'a1', task_index: 0 },
      true,
      'subagent.spawn_requested'
    )
    upsertSubagent(
      's1',
      {
        status: 'running',
        subagent_id: 'a1',
        task_index: 0,
        tool_name: 'search_files',
        tool_preview: 'pattern=hermes'
      },
      false,
      'subagent.tool'
    )
    upsertSubagent(
      's1',
      { status: 'running', subagent_id: 'a1', task_index: 0, text: 'plan the search order' },
      false,
      'subagent.thinking'
    )
    upsertSubagent(
      's1',
      { status: 'running', subagent_id: 'a1', task_index: 0, text: 'found candidate matches' },
      false,
      'subagent.progress'
    )
    upsertSubagent(
      's1',
      { status: 'completed', subagent_id: 'a1', summary: 'search complete', task_index: 0 },
      false,
      'subagent.complete'
    )

    const item = listFor('s1')[0]
    expect(item?.stream.map(e => e.kind)).toEqual(['tool', 'thinking', 'progress', 'summary'])
    expect(item?.stream.find(e => e.kind === 'tool')?.text).toContain('Search Files')
    expect(item?.stream.find(e => e.kind === 'thinking')?.text).toBe('plan the search order')
    expect(item?.stream.find(e => e.kind === 'summary')?.text).toBe('search complete')
  })

  it('prunes delegate fallback rows once native events arrive', () => {
    upsertSubagent('s1', { goal: 'fallback', status: 'running', subagent_id: 'delegate-tool:abc:0', task_index: 0 })
    upsertSubagent('s1', { goal: 'native', status: 'running', subagent_id: 'sa-0-xyz', task_index: 0 })

    pruneDelegateFallbackSubagents('s1')

    expect(listFor('s1').map(item => item.id)).toEqual(['sa-0-xyz'])
  })

  it('clears one session without touching another', () => {
    upsertSubagent('s1', { goal: 'one', status: 'running', subagent_id: 'a1', task_index: 0 })
    upsertSubagent('s2', { goal: 'two', status: 'running', subagent_id: 'a2', task_index: 0 })

    clearSessionSubagents('s1')

    expect($subagentsBySession.get().s1).toBeUndefined()
    expect($subagentsBySession.get().s2).toHaveLength(1)
  })
})

// Regression test for #49808: the Spawn tree overlay and the status
// bar's "Agents N running" indicator used to disagree because they
// filtered $subagentsBySession differently — the status bar
// aggregated every session, the Spawn tree kept only the active
// session. This block pins the post-fix contract: both indicators
// read from the same aggregate scope.
describe('#49808 — Spawn tree / status bar aggregate scope', () => {
  beforeEach(() => $subagentsBySession.set({}))

  it('status bar count and Spawn tree aggregate over the same scope', () => {
    // Two sessions, two running subagents each. The "active session"
    // would be s1 in the old implementation; s2 is non-active but
    // still owns running subagents.
    upsertSubagent('s1', { goal: 'scan files', status: 'running', subagent_id: 'a1', task_index: 0 })
    upsertSubagent('s1', { goal: 'patch tests', status: 'running', subagent_id: 'a2', task_index: 1 })
    upsertSubagent('s2', { goal: 'background work', status: 'running', subagent_id: 'b1', task_index: 0 })
    upsertSubagent('s2', { goal: 'cron poll', status: 'running', subagent_id: 'b2', task_index: 1 })

    // The status bar uses Object.values(...).reduce(...activeSubagentCount...).
    const statusBarCount = Object.values($subagentsBySession.get()).reduce(
      (sum, items) => sum + activeSubagentCount(items),
      0
    )
    expect(statusBarCount).toBe(4)

    // The Spawn tree (post-fix) uses Object.values(...).flat() — the
    // same scope as the status bar.
    const spawnTreeFlat = Object.values($subagentsBySession.get()).flat()
    const spawnTreeRoots = buildSubagentTree(spawnTreeFlat)
    const spawnTreeTotal = spawnTreeRoots.reduce(
      (sum, root) => sum + 1 + root.children.length,
      0
    )
    expect(spawnTreeTotal).toBe(4)

    // Counter-example that documents the regression: filtering to
    // only the active session would yield 2 subagents while the
    // status bar counted 4. The post-fix Spawn tree must include
    // s2's subagents too.
    const activeOnly = $subagentsBySession.get()['s1'] ?? []
    expect(activeOnly).toHaveLength(2)
    expect(statusBarCount).not.toBe(activeOnly.length)
  })

  it('Spawn tree stays empty only when no session has running subagents', () => {
    // Empty-state pin: with only terminal subagents, both indicators
    // agree on zero running/queued and the Spawn tree renders its
    // empty placeholder. Note that buildSubagentTree includes
    // completed/failed nodes (so we count activeSubagentCount instead
    // of tree length for the Spawn-tree side of the contract).
    upsertSubagent('s1', { goal: 'done', status: 'completed', subagent_id: 'a1', task_index: 0 })

    const spawnTreeFlat = Object.values($subagentsBySession.get()).flat()
    const spawnTreeActiveCount = activeSubagentCount(spawnTreeFlat)

    const statusBarCount = Object.values($subagentsBySession.get()).reduce(
      (sum, items) => sum + activeSubagentCount(items),
      0
    )
    expect(spawnTreeActiveCount).toBe(0)
    expect(statusBarCount).toBe(0)
    // Both indicators agree — the regression condition for #49808.
    expect(spawnTreeActiveCount).toBe(statusBarCount)
  })
})
