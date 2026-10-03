import { describe, expect, it } from 'vitest'

import type { SubagentProgress } from '@/store/subagents'

import { delegateGoals, delegateRowsFromCall, mergeDelegateRows } from './delegate-model'

const subagent = (overrides: Partial<SubagentProgress>): SubagentProgress => ({
  filesRead: [],
  filesWritten: [],
  goal: 'Research Cursor',
  id: 'sub-1',
  parentId: null,
  startedAt: 0,
  status: 'running',
  stream: [],
  taskCount: 1,
  taskIndex: 0,
  updatedAt: 0,
  ...overrides
})

describe('delegateGoals', () => {
  it('reads a batch in task order and a single goal alike', () => {
    expect(delegateGoals({ tasks: [{ goal: 'A' }, { goal: 'B' }] })).toEqual(['A', 'B'])
    expect(delegateGoals({ goal: 'Solo' })).toEqual(['Solo'])
    expect(delegateGoals('{"goal":"Serialized"}')).toEqual(['Serialized'])
  })
})

describe('delegateRowsFromCall', () => {
  it('reads as running before a result and parked once dispatched', () => {
    const args = { tasks: [{ goal: 'A' }, { goal: 'B' }] }

    expect(delegateRowsFromCall(args, undefined).map(r => r.status)).toEqual(['running', 'running'])
    expect(delegateRowsFromCall(args, { status: 'dispatched', goals: ['A', 'B'] }).map(r => r.status)).toEqual([
      'dispatched',
      'dispatched'
    ])
  })

  it('takes status, model and duration from each settled result', () => {
    const rows = delegateRowsFromCall(
      { tasks: [{ goal: 'A' }, { goal: 'B' }] },
      {
        results: [
          { status: 'completed', summary: 'found it', model: 'anthropic/claude-opus-5', duration_seconds: 12 },
          { status: 'failed', summary: 'nope' }
        ]
      }
    )

    expect(rows.map(r => r.status)).toEqual(['completed', 'failed'])
    expect(rows[0]).toMatchObject({ activity: ['found it'], durationSeconds: 12, model: 'anthropic/claude-opus-5' })
  })

  // #73728 / #85492: the delegate tool settles rows with 'ok', 'error' or
  // 'timeout' — anything that is not a success must render as failed instead
  // of hiding behind a green 'completed' check.
  it('renders timeout/error settled results as failed, ok as completed', () => {
    const rows = delegateRowsFromCall(
      { tasks: [{ goal: 'A' }, { goal: 'B' }, { goal: 'C' }, { goal: 'D' }] },
      {
        results: [
          { status: 'ok', summary: 'done' },
          { status: 'timeout', error: 'Timed out after 600s' },
          { status: 'error', error: 'boom' },
          { status: 'failure' }
        ]
      }
    )

    expect(rows.map(r => r.status)).toEqual(['completed', 'failed', 'failed', 'failed'])
  })

  it('still lists a background dispatch whose goals only survive in the result', () => {
    expect(delegateRowsFromCall({}, { status: 'dispatched', goals: ['A', 'B'] }).map(r => r.goal)).toEqual(['A', 'B'])
  })
})

describe('mergeDelegateRows', () => {
  it('joins fallback rows by the tool call id they were keyed with', () => {
    const rows = delegateRowsFromCall({ tasks: [{ goal: 'A' }, { goal: 'B' }] }, undefined, 'call-7')

    const merged = mergeDelegateRows(
      rows,
      [
        subagent({ id: 'delegate-tool:call-7:1', goal: 'B', status: 'completed' }),
        subagent({ id: 'delegate-tool:call-7:0', goal: 'A', model: 'gpt-5' })
      ],
      'call-7'
    )

    expect(merged.map(r => r.status)).toEqual(['running', 'completed'])
    expect(merged[0]!.model).toBe('gpt-5')
  })

  it('joins native events by goal text and prefers their live state', () => {
    const rows = delegateRowsFromCall({ tasks: [{ goal: 'Research Cursor' }] }, undefined, 'call-1')

    const merged = mergeDelegateRows(
      rows,
      [
        subagent({
          goal: 'Research Cursor',
          model: 'anthropic/claude-opus-5',
          sessionId: 'child-1',
          stream: [
            { at: 1, kind: 'tool', text: 'Read File("a.ts")' },
            { at: 2, kind: 'progress', text: 'comparing' }
          ]
        })
      ],
      'call-1'
    )

    expect(merged[0]).toMatchObject({
      activity: ['Read File("a.ts")', 'comparing'],
      model: 'anthropic/claude-opus-5',
      sessionId: 'child-1',
      status: 'running'
    })
  })

  it('never lets a second delegation claim another call\u2019s workers', () => {
    const rows = delegateRowsFromCall({ tasks: [{ goal: 'C' }] }, undefined, 'call-2')

    // Two unrelated children in the session, neither matching this call's goal.
    const merged = mergeDelegateRows(
      rows,
      [subagent({ id: 'other-a', goal: 'A' }), subagent({ id: 'other-b', goal: 'B' })],
      'call-2'
    )

    expect(merged[0]!.goal).toBe('C')
    expect(merged[0]!.model).toBeUndefined()
  })

  it('does not infer ownership from matching cardinality and task index', () => {
    const rows = delegateRowsFromCall({ tasks: [{ goal: 'A' }, { goal: 'B' }] }, undefined, 'call-3')

    const merged = mergeDelegateRows(
      rows,
      [
        subagent({ id: 'x', goal: 'renamed A', taskIndex: 0, model: 'm0' }),
        subagent({ id: 'y', goal: 'renamed B', taskIndex: 1, model: 'm1' })
      ],
      'call-3'
    )

    expect(merged).toEqual(rows)
  })

  it('fails closed when multiple running rows share one legacy goal', () => {
    const rows = delegateRowsFromCall(
      { tasks: [{ goal: 'Same legacy goal' }, { goal: 'Same legacy goal' }] },
      undefined,
      'call-duplicate-goal'
    )

    const merged = mergeDelegateRows(
      rows,
      [subagent({ id: 'native-child', goal: 'Same legacy goal', model: 'must-not-leak' })],
      'call-duplicate-goal'
    )

    expect(merged).toEqual(rows)
  })

  it('keeps settled historical calls isolated from the only post-prune live child', () => {
    const calls = [
      { callId: 'call-a', goal: 'Historical A', id: 'sa-a' },
      { callId: 'call-b', goal: 'Historical B', id: 'sa-b' },
      { callId: 'call-c', goal: 'Historical C', id: 'sa-c' },
      { callId: 'call-d', goal: 'Live D', id: 'sa-d' }
    ]

    const live = [
      subagent({
        id: 'sa-d',
        goal: 'Live D',
        model: 'anthropic/claude-opus-5',
        durationSeconds: 9,
        sessionId: 'child-d',
        stream: [{ at: 1, kind: 'tool', text: 'Terminal("C-terminal-activity")' }],
        taskIndex: 0
      })
    ]

    const merged = calls.map(({ callId, goal, id }) =>
      mergeDelegateRows(
        delegateRowsFromCall(
          { tasks: [{ goal }] },
          { status: 'dispatched', goals: [goal], subagent_ids: [id] },
          callId
        ),
        live,
        callId
      )[0]!
    )

    for (const [index, historical] of merged.slice(0, 3).entries()) {
      expect(historical).toMatchObject({
        activity: [],
        goal: calls[index]!.goal,
        id: `${calls[index]!.callId}:0`,
        subagentId: calls[index]!.id,
        status: 'dispatched'
      })
      expect(historical.model).toBeUndefined()
    }

    expect(merged[3]).toMatchObject({
      activity: ['Terminal("C-terminal-activity")'],
      durationSeconds: 9,
      goal: 'Live D',
      id: 'sa-d',
      model: 'anthropic/claude-opus-5',
      sessionId: 'child-d',
      subagentId: 'sa-d',
      status: 'running'
    })
  })

  it('does not alias settled calls that have identical goals but different persisted ids', () => {
    const goal = 'Same visible goal'
    const live = [subagent({ id: 'sa-new', goal, model: 'new-model', stream: [{ at: 1, kind: 'progress', text: 'new activity' }] })]

    const oldRows = delegateRowsFromCall(
      { tasks: [{ goal }] },
      { status: 'dispatched', goals: [goal], subagent_ids: ['sa-old'] },
      'call-old'
    )

    const newRows = delegateRowsFromCall(
      { tasks: [{ goal }] },
      { status: 'dispatched', goals: [goal], subagent_ids: ['sa-new'] },
      'call-new'
    )

    const historical = mergeDelegateRows(oldRows, live, 'call-old')[0]!
    const owner = mergeDelegateRows(newRows, live, 'call-new')[0]!

    expect(historical).toMatchObject({ activity: [], goal, id: 'call-old:0', status: 'dispatched', subagentId: 'sa-old' })
    expect(historical.model).toBeUndefined()
    expect(owner).toMatchObject({
      activity: ['new activity'],
      goal,
      id: 'sa-new',
      model: 'new-model',
      status: 'running',
      subagentId: 'sa-new'
    })
  })

  it('uses sparse persisted ids by task index without shifting the live owner', () => {
    const rows = delegateRowsFromCall(
      { tasks: [{ goal: 'First' }, { goal: 'Second' }, { goal: 'Third' }] },
      { status: 'dispatched', goals: ['First', 'Second', 'Third'], subagent_ids: ['sa-0-first', null, 'sa-2-third'] },
      'call-sparse'
    )

    const merged = mergeDelegateRows(
      rows,
      [subagent({ id: 'sa-2-third', goal: 'Third live', model: 'third-model', taskIndex: 0 })],
      'call-sparse'
    )

    expect(merged[0]).toMatchObject({
      activity: [],
      goal: 'First',
      id: 'call-sparse:0',
      status: 'dispatched',
      subagentId: 'sa-0-first'
    })
    expect(merged[1]).toMatchObject({ activity: [], goal: 'Second', id: 'call-sparse:1', status: 'dispatched' })
    expect(merged[2]).toMatchObject({
      goal: 'Third live',
      id: 'sa-2-third',
      model: 'third-model',
      status: 'running',
      subagentId: 'sa-2-third'
    })
  })

  it('keeps concurrent rows paired when ordered persisted ids and live activity agree', () => {
    const rows = delegateRowsFromCall(
      { tasks: [{ goal: 'Concurrent A' }, { goal: 'Concurrent B' }] },
      { status: 'dispatched', goals: ['Concurrent A', 'Concurrent B'], subagent_ids: ['sa-concurrent-a', 'sa-concurrent-b'] },
      'call-concurrent'
    )

    const merged = mergeDelegateRows(
      rows,
      [
        subagent({ id: 'sa-concurrent-a', goal: 'Concurrent A', taskIndex: 0, stream: [{ at: 1, kind: 'progress', text: 'activity A' }] }),
        subagent({ id: 'sa-concurrent-b', goal: 'Concurrent B', taskIndex: 1, stream: [{ at: 2, kind: 'progress', text: 'activity B' }] })
      ],
      'call-concurrent'
    )

    expect(merged).toMatchObject([
      {
        activity: ['activity A'],
        goal: 'Concurrent A',
        id: 'sa-concurrent-a',
        subagentId: 'sa-concurrent-a'
      },
      {
        activity: ['activity B'],
        goal: 'Concurrent B',
        id: 'sa-concurrent-b',
        subagentId: 'sa-concurrent-b'
      }
    ])
  })

  it('accepts an event-time ownership promotion for a parked legacy call', () => {
    const rows = delegateRowsFromCall(
      { tasks: [{ goal: 'Legacy child' }] },
      { status: 'dispatched', goals: ['Legacy child'] },
      'legacy-call'
    )

    const merged = mergeDelegateRows(
      rows,
      [
        subagent({
          delegateCallId: 'legacy-call',
          delegateRowIndex: 0,
          goal: 'Legacy child',
          id: 'native-child',
          model: 'native-model'
        })
      ],
      'legacy-call'
    )

    expect(merged[0]).toMatchObject({
      goal: 'Legacy child',
      id: 'native-child',
      model: 'native-model',
      subagentId: 'native-child'
    })
  })

  it('renders a promoted native child only in its exact legacy row', () => {
    const rows = delegateRowsFromCall(
      { tasks: [{ goal: 'First legacy child' }, { goal: 'Second legacy child' }] },
      { status: 'dispatched', goals: ['First legacy child', 'Second legacy child'] },
      'legacy-batch'
    )

    const promotedSecond = subagent({
      delegateCallId: 'legacy-batch',
      delegateRowIndex: 1,
      goal: 'Second legacy child',
      id: 'native-second',
      model: 'second-model',
      taskIndex: 1
    })

    const merged = mergeDelegateRows(rows, [promotedSecond], 'legacy-batch')

    expect(merged[0]).toMatchObject({
      activity: [],
      goal: 'First legacy child',
      id: 'legacy-batch:0',
      status: 'dispatched'
    })
    expect(merged[0]!.model).toBeUndefined()
    expect(merged[1]).toMatchObject({
      goal: 'Second legacy child',
      id: 'native-second',
      model: 'second-model',
      subagentId: 'native-second'
    })

    const missingIndex = mergeDelegateRows(
      rows,
      [subagent({ delegateCallId: 'legacy-batch', goal: 'Second legacy child', id: 'native-without-index' })],
      'legacy-batch'
    )

    expect(missingIndex).toEqual(rows)
  })
})
