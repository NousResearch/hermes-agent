import { describe, expect, it } from 'vitest'

import {
  applyPatch,
  dropPatch,
  groupSwimlanes,
  type LaneDimension,
  laneKey,
  lanePreset,
  NO_LANE,
  normalizeSwimlaneBy,
  sendDropPatch
} from './swimlanes'
import type { KanbanBoard, KanbanTask } from './types'

const DIMS: LaneDimension[] = ['project', 'assignee', 'tenant', 'priority']

const task = (id: string, status: string, extra: Partial<KanbanTask> = {}): KanbanTask => ({
  id,
  title: id,
  status,
  ...extra
})

const board = (tasks: KanbanTask[]): KanbanBoard => {
  const names = ['triage', 'ready', 'running', 'done']

  return {
    columns: names.map(name => ({ name, tasks: tasks.filter(t => t.status === name) })),
    tenants: [],
    assignees: [],
    latest_event_id: 0,
    now: 0
  }
}

const TASKS = [
  task('a', 'ready', { project_id: 'p2', assignee: 'zed', tenant: 't1', priority: 2 }),
  task('b', 'ready', { project_id: 'p1', assignee: 'amy', priority: 0 }),
  task('c', 'running', { assignee: 'amy', tenant: 't1', priority: 5 }),
  task('d', 'done', { project_id: 'p2', priority: 2 }),
  task('e', 'ready', { project_id: 'p2', assignee: 'zed', tenant: 't2', priority: 2 })
]

describe('groupSwimlanes', () => {
  it.each(DIMS)('partitions every card into exactly one %s lane, in its own column and order', by => {
    const { columns } = board(TASKS)
    const lanes = groupSwimlanes(columns, by)

    // Lanes exist only when occupied, and the no-value lane (if any) is last.
    expect(lanes.every(lane => lane.count > 0)).toBe(true)
    expect(lanes.findIndex(lane => lane.key === NO_LANE)).toBeOneOf([-1, lanes.length - 1])

    for (const [index, column] of columns.entries()) {
      const merged = lanes.flatMap(lane => lane.columns[index].tasks)

      // Same set as the column, each card in the lane its own key names...
      expect(merged.map(t => t.id).sort()).toEqual(column.tasks.map(t => t.id).sort())

      for (const lane of lanes) {
        const ids = lane.columns[index].tasks.map(t => t.id)

        expect(lane.columns[index].tasks.every(t => laneKey(by, t) === lane.key)).toBe(true)
        // ...and in the column's original relative order.
        expect(ids).toEqual(column.tasks.filter(t => ids.includes(t.id)).map(t => t.id))
      }
    }
  })

  it('seeded lanes exist with no cards, ordered with the occupied ones', () => {
    const unlinked = board(TASKS.map(t => ({ ...t, project_id: null })))
    const names: Record<string, string> = { p1: 'Zeta', p2: 'Alpha', p3: 'Mid' }
    const lanes = groupSwimlanes(unlinked.columns, 'project', key => names[key] ?? key, ['p1', 'p2', 'p3'])

    expect(lanes.map(l => [l.key, l.count])).toEqual([
      ['p2', 0],
      ['p3', 0],
      ['p1', 0],
      [NO_LANE, TASKS.length]
    ])
  })

  it('orders named lanes by display label and priority lanes high to low', () => {
    const { columns } = board(TASKS)
    const names: Record<string, string> = { p1: 'Zeta', p2: 'Alpha' }

    expect(groupSwimlanes(columns, 'project', key => names[key] ?? key).map(l => l.key)).toEqual(['p2', 'p1', NO_LANE])
    expect(groupSwimlanes(columns, 'priority').map(l => l.key)).toEqual(['5', '2', '0'])
  })
})

describe('dropPatch', () => {
  const ready = TASKS[0]

  it('writes only what the drop changes', () => {
    expect(dropPatch('assignee', ready, 'zed', 'ready')).toEqual({})
    expect(dropPatch('assignee', ready, 'zed', 'done')).toEqual({ status: 'done' })
    expect(dropPatch('none', ready, null, 'done')).toEqual({ status: 'done' })
  })

  it('refuses cross-lane drops a PATCH cannot apply', () => {
    expect(dropPatch('project', ready, 'p1', 'ready')).toBeNull()
    expect(dropPatch('tenant', ready, NO_LANE, 'ready')).toBeNull()
    // A claimed card can't be reassigned until reclaimed.
    expect(dropPatch('assignee', TASKS[2], 'zed', 'ready')).toBeNull()
  })

  it.each([
    ['assignee', 'amy'],
    ['assignee', NO_LANE],
    ['priority', '5']
  ] as const)('a %s drop lands the card in the target cell after the optimistic edit', (by, toLane) => {
    const before = board(TASKS)
    const patch = dropPatch(by, ready, toLane, 'triage')

    expect(patch).not.toBeNull()
    const after = applyPatch(before, ready.id, patch!)
    const lane = groupSwimlanes(after.columns, by).find(l => l.key === toLane)
    const triage = after.columns.findIndex(c => c.name === 'triage')

    expect(lane?.columns[triage].tasks.map(t => t.id)).toContain(ready.id)
    expect(after.columns.flatMap(c => c.tasks)).toHaveLength(TASKS.length)
  })
})

describe('sendDropPatch', () => {
  it('a refused status move leaves the lane field unwritten', async () => {
    const sent: object[] = []

    const send = async (part: object) => {
      sent.push(part)

      if ('status' in part) {
        throw new Error('409: unfinished parent')
      }
    }

    await expect(sendDropPatch({ status: 'done', assignee: 'amy' }, send)).rejects.toThrow('409')
    expect(sent).toEqual([{ status: 'done' }])
  })

  it('an accepted move writes the status, then the lane field', async () => {
    const sent: object[] = []

    await sendDropPatch({ status: 'done', assignee: 'amy' }, async part => void sent.push(part))
    await sendDropPatch({ priority: 5 }, async part => void sent.push(part))

    expect(sent).toEqual([{ status: 'done' }, { assignee: 'amy' }, { priority: 5 }])
  })
})

describe('lanePreset', () => {
  it('the no-tenant lane sends no tenant (an empty string would be a tenant of its own)', () => {
    expect(lanePreset('tenant', NO_LANE)).toEqual({})
  })

  it.each(DIMS)('a task created with a %s lane preset belongs to that lane', by => {
    for (const key of new Set(TASKS.map(t => laneKey(by, t)))) {
      const preset = lanePreset(by, key)
      const created = task('new', 'ready', { ...preset, priority: preset.priority ?? 0 })

      expect(laneKey(by, created)).toBe(key)
    }
  })
})

it('falls back to a flat board for an unknown stored grouping', () => {
  expect(normalizeSwimlaneBy('epic')).toBe('none')
  expect(normalizeSwimlaneBy('project')).toBe('project')
})
