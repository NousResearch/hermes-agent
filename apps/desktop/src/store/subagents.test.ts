import { beforeEach, describe, expect, it, vi } from 'vitest'

import {
  $subagentsBySession,
  activeSubagentCount,
  allSubagents,
  buildSubagentTree,
  clearSessionSubagents,
  consumeSessionSubagentHandoffs,
  failedSubagentCount,
  hasDetachedSessionSubagents,
  markSessionSubagentsDetached,
  preserveDetachedSessionSubagents,
  pruneDelegateFallbackSubagents,
  pruneSettledSessionSubagents,
  reconcileProfileSubagents,
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

  it('keeps active reviewers across a new parent turn while pruning settled reviewers', () => {
    upsertSubagent('s1', { goal: 'active', status: 'running', subagent_id: 'active' })
    upsertSubagent('s1', { goal: 'done', status: 'completed', subagent_id: 'done' })

    expect(pruneSettledSessionSubagents('s1')).toBe(true)
    expect(listFor('s1').map(item => item.id)).toEqual(['active'])

    upsertSubagent('s1', { goal: 'active', status: 'completed', subagent_id: 'active' })

    expect(pruneSettledSessionSubagents('s1')).toBe(false)
    expect($subagentsBySession.get().s1).toBeUndefined()
  })

  it('preserves detached reviewers across stop and leases their terminal handoff', () => {
    upsertSubagent('s1', { goal: 'async review', status: 'running', subagent_id: 'async' })
    upsertSubagent('s1', { goal: 'foreground child', status: 'completed', subagent_id: 'done' })

    expect(markSessionSubagentsDetached('s1')).toBe(true)
    expect(hasDetachedSessionSubagents('s1')).toBe(true)
    expect(preserveDetachedSessionSubagents('s1')).toBe(true)
    expect(listFor('s1').map(item => item.id)).toEqual(['async'])

    const completed = upsertSubagent(
      's1',
      { goal: 'async review', status: 'completed', subagent_id: 'async' },
      false,
      'subagent.complete'
    )

    expect(completed?.handoff).toBe(true)
    expect(hasDetachedSessionSubagents('s1')).toBe(true)
    expect(pruneSettledSessionSubagents('s1')).toBe(true)
    expect(listFor('s1').map(item => item.id)).toEqual(['async'])
    expect(pruneSettledSessionSubagents('s1', true)).toBe(false)
    expect($subagentsBySession.get().s1).toBeUndefined()
  })

  it('classifies background children at spawn and preserves a fast terminal handoff', () => {
    upsertSubagent('s1', { detached: true, goal: 'fast async', status: 'running', subagent_id: 'fast' })
    upsertSubagent(
      's1',
      { detached: true, goal: 'fast async', status: 'completed', subagent_id: 'fast' },
      false,
      'subagent.complete'
    )

    expect(listFor('s1')[0]).toMatchObject({ detached: true, handoff: true, status: 'completed' })
    expect(preserveDetachedSessionSubagents('s1')).toBe(true)
    expect(pruneSettledSessionSubagents('s1')).toBe(true)
    expect(pruneSettledSessionSubagents('s1', true)).toBe(false)
  })

  it('consumes only the terminal handoff named by one async delivery', () => {
    for (const id of ['batch-a', 'batch-b']) {
      upsertSubagent('s1', { detached: true, goal: id, status: 'running', subagent_id: id }, true, '', 's1', 'alpha')
      upsertSubagent(
        's1',
        { detached: true, goal: id, status: 'completed', subagent_id: id },
        false,
        'subagent.complete',
        's1',
        'alpha'
      )
    }

    expect(consumeSessionSubagentHandoffs('s1', ['batch-a'], 'alpha')).toBe(true)
    expect(listFor('s1').map(item => item.id)).toEqual(['batch-b'])
    expect(consumeSessionSubagentHandoffs('s1', ['batch-b'], 'alpha')).toBe(true)
    expect($subagentsBySession.get().s1).toBeUndefined()
  })

  it('keeps identical runtime and subagent ids isolated by profile', () => {
    upsertSubagent(
      'same-runtime',
      { goal: 'alpha', status: 'running', subagent_id: 'same-child' },
      true,
      '',
      'a',
      'alpha'
    )
    upsertSubagent(
      'same-runtime',
      { goal: 'beta', status: 'running', subagent_id: 'same-child' },
      true,
      '',
      'b',
      'beta'
    )

    expect(listFor('same-runtime')).toMatchObject([
      { goal: 'alpha', ownerSessionId: 'a', profile: 'alpha' },
      { goal: 'beta', ownerSessionId: 'b', profile: 'beta' }
    ])

    preserveDetachedSessionSubagents('same-runtime', 'alpha')

    expect(listFor('same-runtime')).toMatchObject([{ goal: 'beta', profile: 'beta' }])
  })

  it('reconciles missed starts and terminals without touching another profile', () => {
    upsertSubagent(
      'runtime-a',
      { detached: true, goal: 'stale alpha', status: 'running', subagent_id: 'stale' },
      true,
      '',
      'owner-a',
      'alpha'
    )
    upsertSubagent(
      'runtime-b',
      { detached: true, goal: 'live beta', status: 'running', subagent_id: 'beta' },
      true,
      '',
      'owner-b',
      'beta'
    )

    reconcileProfileSubagents('alpha', [
      {
        detached: true,
        goal: 'recovered alpha',
        owner_session_id: 'owner-a',
        status: 'running',
        subagent_id: 'recovered'
      }
    ])

    const all = allSubagents($subagentsBySession.get())
    expect(all.map(item => `${item.profile}:${item.id}`).sort()).toEqual(['alpha:recovered', 'beta:beta'])
  })

  it('restores a completed detached handoff that is still awaiting delivery', () => {
    reconcileProfileSubagents('alpha', [
      {
        detached: true,
        goal: 'completed review',
        handoff: true,
        owner_session_id: 'owner-a',
        status: 'completed',
        subagent_id: 'pending'
      }
    ])

    expect(listFor('owner-a')).toMatchObject([
      { detached: true, handoff: true, id: 'pending', profile: 'alpha', status: 'completed' }
    ])
  })

  it('does not let a slow snapshot prune an event received after the request started', () => {
    const nowSpy = vi.spyOn(Date, 'now').mockReturnValue(2_000)
    upsertSubagent(
      'runtime-a',
      { goal: 'new event', status: 'running', subagent_id: 'new' },
      true,
      '',
      'owner-a',
      'alpha'
    )
    nowSpy.mockRestore()

    reconcileProfileSubagents('alpha', [], 1_500)
    expect(listFor('runtime-a')).toMatchObject([{ id: 'new', profile: 'alpha' }])

    reconcileProfileSubagents('alpha', [], 2_500)
    expect($subagentsBySession.get()['runtime-a']).toBeUndefined()
  })

  it('drops non-detached children when a foreground turn is stopped', () => {
    upsertSubagent('s1', { goal: 'foreground child', status: 'running', subagent_id: 'sync' })

    expect(preserveDetachedSessionSubagents('s1')).toBe(false)
    expect($subagentsBySession.get().s1).toBeUndefined()
  })

  it.each(['error', 'timeout'])('treats backend %s completions as terminal failures', status => {
    upsertSubagent('s1', { goal: 'review', status: 'running', subagent_id: status })
    upsertSubagent('s1', { goal: 'review', status, subagent_id: status }, false, 'subagent.complete')

    expect(listFor('s1')[0]?.status).toBe('failed')
    expect(activeSubagentCount(listFor('s1'))).toBe(0)
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
