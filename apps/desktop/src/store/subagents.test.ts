import { beforeEach, describe, expect, it, vi } from 'vitest'

import {
  $subagentsBySession,
  activeSubagentCount,
  allSubagents,
  buildSubagentTree,
  clearSessionSubagents,
  failedSubagentCount,
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

  // Contract: the status-bar "Agents" indicator and the Spawn-tree panel read
  // the same scope — every session's subagents — so a count can never point at
  // an empty tree (the desync behind "Agents (N)" vs "No live subagents").
  it('counts running/failed across every session, matching the aggregated tree', () => {
    upsertSubagent('s1', { goal: 'a', status: 'running', subagent_id: 'a', task_index: 0 })
    upsertSubagent('s1', { goal: 'b', status: 'failed', subagent_id: 'b', task_index: 1 })
    upsertSubagent('s2', { goal: 'c', status: 'running', subagent_id: 'c', task_index: 0 })
    upsertSubagent('s2', { goal: 'd', status: 'failed', subagent_id: 'd', task_index: 1 })

    const flat = allSubagents($subagentsBySession.get())
    const indicatorRunning = Object.values($subagentsBySession.get()).reduce((n, l) => n + activeSubagentCount(l), 0)
    const indicatorFailed = Object.values($subagentsBySession.get()).reduce((n, l) => n + failedSubagentCount(l), 0)
    const tree = buildSubagentTree(flat)

    // The active-session-only filter would have reported 1/1 here, not 2/2.
    expect(indicatorRunning).toBe(2)
    expect(indicatorFailed).toBe(2)
    expect(tree).toHaveLength(4)
    expect(indicatorRunning + indicatorFailed).toBe(tree.length)
  })

  it('clears one session without touching another', () => {
    upsertSubagent('s1', { goal: 'one', status: 'running', subagent_id: 'a1', task_index: 0 })
    upsertSubagent('s2', { goal: 'two', status: 'running', subagent_id: 'a2', task_index: 0 })

    clearSessionSubagents('s1')

    expect($subagentsBySession.get().s1).toBeUndefined()
    expect($subagentsBySession.get().s2).toHaveLength(1)
  })
})
