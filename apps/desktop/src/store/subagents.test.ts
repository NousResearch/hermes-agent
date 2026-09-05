import { beforeEach, describe, expect, it, vi } from 'vitest'

import {
  $subagentsBySession,
  activeSubagentCount,
  activeSubagentsForStoredSession,
  allSubagents,
  buildSubagentTree,
  clearSessionSubagents,
  failedSubagentCount,
  pruneDelegateFallbackSubagents,
  pruneFinishedSessionSubagents,
  restoreActiveSubagents,
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

  // Regression test for #64015: still-RUNNING background subagents must survive
  // the per-turn wipe that previously dropped them at message.start. The fix
  // replaces clearSessionSubagents() with pruneFinishedSessionSubagents() at
  // the use-message-stream message.start handler, so only terminal-status rows
  // get filtered out.
  it('pruneFinishedSessionSubagents keeps running/queued and drops terminal rows', () => {
    upsertSubagent('s1', { goal: 'live-a', status: 'running', subagent_id: 'live-a', task_index: 0 })
    upsertSubagent('s1', { goal: 'live-b', status: 'queued', subagent_id: 'live-b', task_index: 1 })
    upsertSubagent('s1', { goal: 'done', status: 'completed', subagent_id: 'done', task_index: 2 })
    upsertSubagent('s1', { goal: 'broken', status: 'failed', subagent_id: 'broken', task_index: 3 })
    upsertSubagent('s1', { goal: 'cancelled', status: 'interrupted', subagent_id: 'cancelled', task_index: 4 })

    pruneFinishedSessionSubagents('s1')

    const ids = listFor('s1')
      .map(item => item.id)
      .sort()

    expect(ids).toEqual(['live-a', 'live-b'])
    expect(activeSubagentCount(listFor('s1'))).toBe(2)
  })

  // Companion test: after prune, a late `subagent.complete` event for a
  // surviving live row must still be accepted by upsertSubagent (the wipe
  // path previously silently dropped these).
  it('surviving live subagents still accept createIfMissing=false completion', () => {
    upsertSubagent('s1', { goal: 'live', status: 'running', subagent_id: 'live', task_index: 0 })

    pruneFinishedSessionSubagents('s1')

    upsertSubagent(
      's1',
      { status: 'completed', subagent_id: 'live', task_index: 0, summary: 'finished later' },
      false,
      'subagent.complete'
    )

    const item = listFor('s1')[0]
    expect(item?.status).toBe('completed')
    expect(item?.summary).toBe('finished later')
  })

  it.each(['completed', 'failed', 'interrupted'] as const)(
    'keeps an authoritative %s row when a stale running snapshot arrives',
    status => {
      upsertSubagent(
        's1',
        { status, subagent_id: 'finished-during-reload', summary: 'finished while reconnecting' },
        false,
        'subagent.complete'
      )

      expect(listFor('s1')).toMatchObject([
        {
          id: 'finished-during-reload',
          status,
          summary: 'finished while reconnecting'
        }
      ])

      restoreActiveSubagents('s1', [
        {
          goal: 'stale running snapshot',
          owner_agent_session_id: 'stored-owner',
          status: 'running',
          subagent_id: 'finished-during-reload'
        }
      ])

      expect(listFor('s1')[0]).toMatchObject({
        id: 'finished-during-reload',
        status,
        summary: 'finished while reconnecting'
      })
    }
  )

  it('pruneFinishedSessionSubagents leaves other sessions untouched', () => {
    upsertSubagent('s1', { goal: 'live', status: 'running', subagent_id: 'a', task_index: 0 })
    upsertSubagent('s1', { goal: 'done', status: 'completed', subagent_id: 'b', task_index: 1 })
    upsertSubagent('s2', { goal: 'live', status: 'running', subagent_id: 'c', task_index: 0 })
    upsertSubagent('s2', { goal: 'done', status: 'completed', subagent_id: 'd', task_index: 1 })

    pruneFinishedSessionSubagents('s1')

    expect(listFor('s1').map(item => item.id)).toEqual(['a'])
    expect(
      listFor('s2')
        .map(item => item.id)
        .sort()
    ).toEqual(['c', 'd'])
  })

  // Regression test for #73728: backend terminal statuses like `timeout` and
  // `error` were normalised to `running`, making timed-out subagents immortal
  // in the active status stack. `cancelled`/`canceled` must also map to
  // `interrupted`.
  it('normalises backend terminal statuses to recognised SubagentStatus values', () => {
    upsertSubagent('s1', { goal: 'a', status: 'running', subagent_id: 'a', task_index: 0 })
    upsertSubagent('s1', { goal: 'b', status: 'running', subagent_id: 'b', task_index: 1 })
    upsertSubagent('s1', { goal: 'c', status: 'running', subagent_id: 'c', task_index: 2 })
    upsertSubagent('s1', { goal: 'd', status: 'running', subagent_id: 'd', task_index: 3 })

    // Emit terminal events with backend-native status strings
    upsertSubagent(
      's1',
      { status: 'timeout', subagent_id: 'a', task_index: 0, summary: 'timed out' },
      false,
      'subagent.complete'
    )
    upsertSubagent(
      's1',
      { status: 'error', subagent_id: 'b', task_index: 1, summary: 'errored' },
      false,
      'subagent.complete'
    )
    upsertSubagent('s1', { status: 'cancelled', subagent_id: 'c', task_index: 2 }, false, 'subagent.complete')
    upsertSubagent('s1', { status: 'canceled', subagent_id: 'd', task_index: 3 }, false, 'subagent.complete')

    const items = listFor('s1')
    const byId = Object.fromEntries(items.map(i => [i.id, i]))

    // timeout → failed
    expect(byId['a']?.status).toBe('failed')
    expect(byId['a']?.currentTool).toBeUndefined()

    // error → failed
    expect(byId['b']?.status).toBe('failed')

    // cancelled → interrupted
    expect(byId['c']?.status).toBe('interrupted')

    // canceled → interrupted
    expect(byId['d']?.status).toBe('interrupted')

    // All four are terminal — prune should remove them all
    pruneFinishedSessionSubagents('s1')
    expect(listFor('s1')).toHaveLength(0)
  })

  // The backend completes subagents with status "timeout" (hard child timeout,
  // delegation.child_timeout_seconds) and no summary — synthesize the reason
  // so the failed row explains itself instead of rendering as a bare failure.
  it('maps backend timeout status to a terminal failure with a synthesized reason', () => {
    upsertSubagent('s1', { goal: 'scan files', status: 'running', subagent_id: 't1', task_index: 0 })
    upsertSubagent(
      's1',
      { status: 'timeout', subagent_id: 't1', task_index: 0, duration_seconds: 612.3 },
      false,
      'subagent.complete'
    )

    const item = listFor('s1')[0]
    expect(item?.status).toBe('failed')
    expect(item?.durationSeconds).toBe(612.3)
    expect(item?.summary).toBe('Timed out after 612.3s')

    // A timed-out row must be pruned at the next message.start boundary like
    // any other finished row — it must not linger as a live spinner.
    pruneFinishedSessionSubagents('s1')
    expect(listFor('s1')).toHaveLength(0)
  })

  it('falls back to a placeholder when timeout duration is missing', () => {
    upsertSubagent('s1', { goal: 'scan files', status: 'running', subagent_id: 't2', task_index: 0 })
    upsertSubagent('s1', { status: 'timeout', subagent_id: 't2', task_index: 0 }, false, 'subagent.complete')

    expect(listFor('s1')[0]?.summary).toBe('Timed out after ?s')
  })

  // Fail-closed guard: subagent.complete is terminal by definition, so an
  // unrecognized status on it must not resurrect a row as 'running'. Live
  // events keep the lenient fallback (a status we don't know is still active).
  it('fails closed on unrecognized completion statuses but stays lenient for live events', () => {
    upsertSubagent('s1', { goal: 'scan files', status: 'running', subagent_id: 'u1', task_index: 0 })
    upsertSubagent(
      's1',
      { status: 'some_future_terminal_status', subagent_id: 'u1', task_index: 0 },
      false,
      'subagent.complete'
    )
    expect(listFor('s1')[0]?.status).toBe('failed')
    expect(activeSubagentCount(listFor('s1'))).toBe(0)

    upsertSubagent('s1', { goal: 'scan files', status: 'running', subagent_id: 'u2', task_index: 1 })
    upsertSubagent(
      's1',
      { status: 'some_future_live_status', subagent_id: 'u2', task_index: 1, text: 'still working' },
      false,
      'subagent.progress'
    )
    expect(listFor('s1')[1]?.status).toBe('running')
    expect(activeSubagentCount(listFor('s1'))).toBe(1)
  })

  // Folded in from PR #85995: a subagent.complete carrying a still-active
  // payload status ('running'/'queued') must also settle as failed — the
  // event itself is the source of truth that the child is done.
  it.each(['running', 'queued'] as const)(
    'treats a completion event with %s payload status as terminal failure',
    status => {
      upsertSubagent(
        's1',
        {
          goal: 'inconsistent completion',
          status: 'running',
          subagent_id: 'ic1',
          task_index: 0,
          tool_name: 'search_files'
        },
        true,
        'subagent.start'
      )
      upsertSubagent('s1', { status, subagent_id: 'ic1', task_index: 0 }, false, 'subagent.complete')

      const items = listFor('s1')
      expect(items[0]?.status).toBe('failed')
      expect(items[0]?.currentTool).toBeUndefined()
      expect(activeSubagentCount(items)).toBe(0)
    }
  )

  // Folded in from PR #80045 (#80018): a late progress event must not revive
  // the spinner after a terminal completion — the row stays settled.
  it('does not regress to running when a late running event arrives after timeout', () => {
    upsertSubagent('s1', { goal: 'task', status: 'running', subagent_id: 'late1', task_index: 0 })
    upsertSubagent(
      's1',
      { goal: 'task', status: 'timeout', subagent_id: 'late1', summary: 'Timed out', task_index: 0 },
      true,
      'subagent.complete'
    )
    upsertSubagent('s1', { goal: 'task', status: 'running', subagent_id: 'late1', task_index: 0, text: 'late' })

    expect(listFor('s1')[0]?.status).toBe('failed')
  })

  it('restores a running backend snapshot into an empty runtime bucket', () => {
    restoreActiveSubagents('runtime-owner', [
      {
        goal: 'inspect the renderer',
        owner_agent_session_id: 'stored-owner',
        started_at: 1_700_000_000,
        status: 'running',
        subagent_id: 'sa-restored'
      }
    ])

    expect(listFor('runtime-owner')).toMatchObject([
      {
        goal: 'inspect the renderer',
        id: 'sa-restored',
        startedAt: 1_700_000_000_000,
        status: 'running'
      }
    ])
  })

  it('selects the durable owner across compression lineage without leaking another session', () => {
    const entries = [
      { owner_agent_session_id: 'root-owner', subagent_id: 'ours' },
      { owner_agent_session_id: 'stored-foreign', subagent_id: 'foreign' },
      { subagent_id: 'unowned' }
    ]

    const sessions = [
      {
        _lineage_ids: ['root-owner', 'middle-owner', 'tip-owner'],
        _lineage_root_id: 'root-owner',
        id: 'tip-owner'
      }
    ]

    expect(activeSubagentsForStoredSession('tip-owner', entries, sessions as never)).toEqual([entries[0]])
  })

  it('keeps nested descendants whose durable owner is their immediate parent session', () => {
    const entries = [
      { owner_agent_session_id: 'stored-owner', subagent_id: 'root-child' },
      {
        owner_agent_session_id: 'root-child-session',
        parent_id: 'root-child',
        subagent_id: 'nested-child'
      },
      {
        owner_agent_session_id: 'nested-child-session',
        parent_id: 'nested-child',
        subagent_id: 'nested-grandchild'
      },
      { owner_agent_session_id: 'stored-foreign', subagent_id: 'foreign-root' }
    ]

    expect(activeSubagentsForStoredSession('stored-owner', entries, [] as never).map(item => item.subagent_id)).toEqual(
      ['root-child', 'nested-child', 'nested-grandchild']
    )
  })
})
