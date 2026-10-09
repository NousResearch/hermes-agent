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

  it('falls back to task order only when both sides agree on the shape', () => {
    const rows = delegateRowsFromCall({ tasks: [{ goal: 'A' }, { goal: 'B' }] }, undefined, 'call-3')

    const merged = mergeDelegateRows(
      rows,
      [
        subagent({ id: 'x', goal: 'renamed A', taskIndex: 0, model: 'm0' }),
        subagent({ id: 'y', goal: 'renamed B', taskIndex: 1, model: 'm1' })
      ],
      'call-3'
    )

    expect(merged.map(r => r.model)).toEqual(['m0', 'm1'])
  })
})
