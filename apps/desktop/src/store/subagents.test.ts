import { beforeEach, describe, expect, it, vi } from 'vitest'

import {
  $subagentsBySession,
  activeSubagentCount,
  allSubagents,
  buildSubagentTree,
  clearSessionSubagents,
  failedSubagentCount,
  pruneDelegateFallbackSubagents,
  pruneFinishedSessionSubagents,
  reconcileSubagentSnapshot,
  upsertSubagent
} from './subagents'
import { subagentEvent } from './subagents.test-util'

const listFor = (sid: string) => $subagentsBySession.get()[sid] ?? []

describe('subagent store', () => {
  beforeEach(() => $subagentsBySession.set({}))

  it('upserts subagent progress and keeps terminal status stable', () => {
    upsertSubagent('s1', subagentEvent({ goal: 'scan files', status: 'running', subagent_id: 'a1' }))
    upsertSubagent('s1', subagentEvent({ goal: 'scan files', status: 'completed', subagent_id: 'a1', summary: 'done' }))
    upsertSubagent('s1', subagentEvent({ goal: 'scan files', status: 'running', subagent_id: 'a1', text: 'late' }))

    const item = listFor('s1')[0]
    expect(item?.status).toBe('completed')
    expect(item?.summary).toBe('done')
  })

  it('keeps completed children retired across turn pruning, late frames, and roster refreshes', () => {
    const finished = { subagent_id: 'finished', goal: 'Finished task', status: 'running' }
    const live = { subagent_id: 'live', goal: 'Background task', status: 'queued' }
    upsertSubagent('owner', finished, true, 'subagent.start')
    upsertSubagent('owner', live, true, 'subagent.spawn_requested')
    upsertSubagent('owner', { ...finished, status: 'completed', summary: 'Done' }, false, 'subagent.complete')
    upsertSubagent('owner', { ...finished, text: '(°□°) pondering...' }, false, 'subagent.thinking')
    expect(listFor('owner')[0]?.status).toBe('completed')

    // A completion starts a new parent turn before every delayed child frame
    // or roster read has drained. Pruning is presentation, not a new child run.
    pruneFinishedSessionSubagents('owner')
    const pruned = listFor('owner')
    reconcileSubagentSnapshot('owner', [finished, live])
    upsertSubagent('owner', finished, true, 'subagent.start')
    upsertSubagent('owner', { ...finished, text: '(°□°) pondering...' }, false, 'subagent.thinking')
    expect(listFor('owner')).toBe(pruned)
    expect(listFor('owner').map(item => item.id)).toEqual(['live'])

    // Roster-discovered terminal state has the same authority as an event.
    reconcileSubagentSnapshot('owner', [{ ...live, status: 'interrupted' }])
    pruneFinishedSessionSubagents('owner')
    reconcileSubagentSnapshot('owner', [finished, live])
    expect(activeSubagentCount(listFor('owner'))).toBe(0)

    // Retirement is scoped to this runtime session, not an ID-global ban.
    upsertSubagent('other', finished, true, 'subagent.start')
    expect(activeSubagentCount(listFor('other'))).toBe(1)
    clearSessionSubagents('owner')
    upsertSubagent('owner', finished, true, 'subagent.start')
    expect(activeSubagentCount(listFor('owner'))).toBe(1)
  })

  it('builds parent/child trees', () => {
    upsertSubagent('s1', subagentEvent({ goal: 'parent', status: 'running', subagent_id: 'p' }))
    upsertSubagent(
      's1',
      subagentEvent({ goal: 'child', parent_id: 'p', status: 'queued', subagent_id: 'c', task_index: 1 })
    )

    const tree = buildSubagentTree(listFor('s1'))
    expect(tree).toHaveLength(1)
    expect(tree[0]?.children[0]?.goal).toBe('child')
    expect(activeSubagentCount(listFor('s1'))).toBe(2)
  })

  it('keeps root nodes in spawn order, not task index order', () => {
    const nowSpy = vi.spyOn(Date, 'now')
    nowSpy.mockReturnValueOnce(1_000)
    upsertSubagent('s1', subagentEvent({ goal: 'first spawn', status: 'running', subagent_id: 'a', task_index: 2 }))
    nowSpy.mockReturnValueOnce(2_000)
    upsertSubagent('s1', subagentEvent({ goal: 'second spawn', status: 'running', subagent_id: 'b' }))
    nowSpy.mockRestore()

    expect(buildSubagentTree(listFor('s1')).map(n => n.id)).toEqual(['a', 'b'])
  })

  it('captures live thinking/progress/tool stream lines', () => {
    upsertSubagent(
      's1',
      subagentEvent({ goal: 'scan files', status: 'queued', subagent_id: 'a1' }),
      true,
      'subagent.spawn_requested'
    )
    upsertSubagent(
      's1',
      subagentEvent({
        status: 'running',
        subagent_id: 'a1',
        tool_name: 'search_files',
        tool_preview: 'pattern=hermes'
      }),
      false,
      'subagent.tool'
    )
    upsertSubagent(
      's1',
      subagentEvent({ status: 'running', subagent_id: 'a1', text: 'plan the search order' }),
      false,
      'subagent.thinking'
    )
    upsertSubagent(
      's1',
      subagentEvent({ status: 'running', subagent_id: 'a1', text: 'found candidate matches' }),
      false,
      'subagent.progress'
    )
    upsertSubagent(
      's1',
      subagentEvent({ status: 'completed', subagent_id: 'a1', summary: 'search complete' }),
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
    upsertSubagent('s1', subagentEvent({ goal: 'fallback', status: 'running', subagent_id: 'delegate-tool:abc:0' }))
    upsertSubagent('s1', subagentEvent({ goal: 'native', status: 'running', subagent_id: 'sa-0-xyz' }))

    pruneDelegateFallbackSubagents('s1')

    expect(listFor('s1').map(item => item.id)).toEqual(['sa-0-xyz'])
  })

  // Contract: the status-bar "Agents" indicator and the Spawn-tree panel read
  // the same scope — every session's subagents — so a count can never point at
  // an empty tree (the desync behind "Agents (N)" vs "No live subagents").
  it('counts running/failed across every session, matching the aggregated tree', () => {
    upsertSubagent('s1', subagentEvent({ goal: 'a', status: 'running', subagent_id: 'a' }))
    upsertSubagent('s1', subagentEvent({ goal: 'b', status: 'failed', subagent_id: 'b', task_index: 1 }))
    upsertSubagent('s2', subagentEvent({ goal: 'c', status: 'running', subagent_id: 'c' }))
    upsertSubagent('s2', subagentEvent({ goal: 'd', status: 'timeout', subagent_id: 'd', task_index: 1 }))

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
    upsertSubagent('s1', subagentEvent({ goal: 'one', status: 'running', subagent_id: 'a1' }))
    upsertSubagent('s2', subagentEvent({ goal: 'two', status: 'running', subagent_id: 'a2' }))

    clearSessionSubagents('s1')

    expect($subagentsBySession.get().s1).toBeUndefined()
    expect($subagentsBySession.get().s2).toHaveLength(1)
  })

  // Regression test for #64015: still-RUNNING background subagents must survive
  // the per-turn wipe that previously dropped them at message.start. The fix
  // replaces clearSessionSubagents() with pruneFinishedSessionSubagents() at
  // the use-message-stream message.start handler, so only terminal-status rows
  // get filtered out.
  it('pruneFinishedSessionSubagents keeps running/queued and drops every terminal status', () => {
    upsertSubagent('s1', subagentEvent({ goal: 'live-a', status: 'running', subagent_id: 'live-a' }))
    upsertSubagent('s1', subagentEvent({ goal: 'live-b', status: 'queued', subagent_id: 'live-b', task_index: 1 }))
    upsertSubagent('s1', subagentEvent({ goal: 'done', status: 'completed', subagent_id: 'done', task_index: 2 }))
    upsertSubagent('s1', subagentEvent({ goal: 'broken', status: 'failed', subagent_id: 'broken', task_index: 3 }))
    upsertSubagent(
      's1',
      subagentEvent({ goal: 'stopped', status: 'interrupted', subagent_id: 'stopped', task_index: 4 })
    )
    upsertSubagent('s1', subagentEvent({ goal: 'errored', status: 'error', subagent_id: 'errored', task_index: 5 }))
    upsertSubagent('s1', subagentEvent({ goal: 'late', status: 'timeout', subagent_id: 'late', task_index: 6 }))

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
    upsertSubagent('s1', subagentEvent({ goal: 'live', status: 'running', subagent_id: 'live' }))

    pruneFinishedSessionSubagents('s1')

    upsertSubagent(
      's1',
      subagentEvent({ status: 'completed', subagent_id: 'live', summary: 'finished later' }),
      false,
      'subagent.complete'
    )

    const item = listFor('s1')[0]
    expect(item?.status).toBe('completed')
    expect(item?.summary).toBe('finished later')
  })

  it('pruneFinishedSessionSubagents leaves other sessions untouched', () => {
    upsertSubagent('s1', subagentEvent({ goal: 'live', status: 'running', subagent_id: 'a' }))
    upsertSubagent('s1', subagentEvent({ goal: 'done', status: 'completed', subagent_id: 'b', task_index: 1 }))
    upsertSubagent('s2', subagentEvent({ goal: 'live', status: 'running', subagent_id: 'c' }))
    upsertSubagent('s2', subagentEvent({ goal: 'done', status: 'completed', subagent_id: 'd', task_index: 1 }))

    pruneFinishedSessionSubagents('s1')

    expect(listFor('s1').map(item => item.id)).toEqual(['a'])
    expect(
      listFor('s2')
        .map(item => item.id)
        .sort()
    ).toEqual(['c', 'd'])
  })

  // #73728: `timeout` and `error` are backend statuses in their own right. The
  // store used to fold them into `failed`, and before that into `running` —
  // which left timed-out children spinning forever in the status stack. They
  // now survive the wire verbatim and stay terminal.
  it.each([
    { status: 'timeout', summary: 'Timed out after 612.3s' },
    { status: 'error', summary: 'boom' }
  ] as const)('settles a $status completion as its own terminal status', ({ status, summary }) => {
    upsertSubagent(
      's1',
      subagentEvent({ goal: 'scan files', status: 'running', subagent_id: 'w', tool_name: 'search_files' }),
      true,
      'subagent.start'
    )
    upsertSubagent(
      's1',
      subagentEvent({
        duration_seconds: 612.3,
        status,
        subagent_id: 'w',
        summary: status === 'error' ? 'boom' : null
      }),
      false,
      'subagent.complete'
    )

    const item = listFor('s1')[0]
    expect(item?.status).toBe(status)
    expect(item?.currentTool).toBeUndefined()
    expect(item?.summary).toBe(summary)
    expect(activeSubagentCount(listFor('s1'))).toBe(0)
    expect(failedSubagentCount(listFor('s1'))).toBe(1)

    // A settled row is pruned at the next message.start boundary like any
    // other finished row — it must not linger as a live spinner.
    pruneFinishedSessionSubagents('s1')
    expect(listFor('s1')).toHaveLength(0)
  })

  it('falls back to a placeholder when timeout duration is missing', () => {
    upsertSubagent('s1', subagentEvent({ goal: 'scan files', status: 'running', subagent_id: 't2' }))
    upsertSubagent('s1', subagentEvent({ status: 'timeout', subagent_id: 't2' }), false, 'subagent.complete')

    expect(listFor('s1')[0]?.summary).toBe('Timed out after ?s')
  })

  // Fail-closed guard: subagent.complete is terminal by definition, so a frame
  // that reports no status must not leave a row spinning. Live events keep the
  // lenient fallback (no status means "carry on with what we knew").
  it('fails closed on a completion with no status but stays lenient for live events', () => {
    upsertSubagent('s1', subagentEvent({ goal: 'scan files', status: 'running', subagent_id: 'u1' }))
    upsertSubagent('s1', subagentEvent({ subagent_id: 'u1' }), false, 'subagent.complete')
    expect(listFor('s1')[0]?.status).toBe('failed')
    expect(activeSubagentCount(listFor('s1'))).toBe(0)

    upsertSubagent('s1', subagentEvent({ goal: 'scan files', status: 'running', subagent_id: 'u2', task_index: 1 }))
    upsertSubagent(
      's1',
      subagentEvent({ subagent_id: 'u2', task_index: 1, text: 'still working' }),
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
        subagentEvent({
          goal: 'inconsistent completion',
          status: 'running',
          subagent_id: 'ic1',
          tool_name: 'search_files'
        }),
        true,
        'subagent.start'
      )
      upsertSubagent('s1', subagentEvent({ status, subagent_id: 'ic1' }), false, 'subagent.complete')

      const items = listFor('s1')
      expect(items[0]?.status).toBe('failed')
      expect(items[0]?.currentTool).toBeUndefined()
      expect(activeSubagentCount(items)).toBe(0)
    }
  )

  // Folded in from PR #80045 (#80018): a late progress event must not revive
  // the spinner after a terminal completion — the row stays settled.
  it('does not regress to running when a late running event arrives after timeout', () => {
    upsertSubagent('s1', subagentEvent({ goal: 'task', status: 'running', subagent_id: 'late1' }))
    upsertSubagent(
      's1',
      subagentEvent({ goal: 'task', status: 'timeout', subagent_id: 'late1', summary: 'Timed out' }),
      true,
      'subagent.complete'
    )
    upsertSubagent('s1', subagentEvent({ goal: 'task', status: 'running', subagent_id: 'late1', text: 'late' }))

    expect(listFor('s1')[0]?.status).toBe('timeout')
  })
})
