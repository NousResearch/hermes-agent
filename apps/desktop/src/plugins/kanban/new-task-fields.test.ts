import { describe, expect, it } from 'vitest'

import { initialAssignee, laneCreateFields, PARKED, submittedAssignee } from './new-task-fields'
import { type LaneDimension, laneKey, lanePreset, NO_LANE } from './swimlanes'
import type { KanbanTask } from './types'

const DEFAULT_PROFILE = 'default'

/** The create-time defaults of `kanban_db.create_task` the dialog relies on:
 *  an omitted project_id adopts a project-scoped board's project ('' opts
 *  out), an omitted tenant inherits the parent's. */
function created(body: Record<string, unknown>, ctx: { boardProject: null | string; parentTenant: null | string }) {
  return {
    id: 't_new',
    title: 'new',
    status: 'ready',
    assignee: (body.assignee as string | undefined) ?? null,
    priority: (body.priority as number | undefined) ?? 0,
    project_id: body.project_id === undefined ? ctx.boardProject : (body.project_id as string) || null,
    tenant: (body.tenant as string | undefined) ?? ctx.parentTenant
  } satisfies KanbanTask
}

/** What the dialog submits when opened from a lane and confirmed untouched. */
function dialogBody(by: LaneDimension, key: string) {
  const preset = lanePreset(by, key)

  return {
    assignee: submittedAssignee(initialAssignee(preset, DEFAULT_PROFILE), DEFAULT_PROFILE),
    priority: preset.priority ?? 0,
    ...laneCreateFields(preset)
  }
}

const LANES: Array<[LaneDimension, string]> = [
  ['project', 'p_atlas'],
  ['project', NO_LANE],
  ['assignee', 'alice'],
  ['assignee', DEFAULT_PROFILE],
  ['assignee', NO_LANE],
  ['tenant', 'acme'],
  ['tenant', NO_LANE],
  ['priority', '3'],
  ['priority', '0']
]

describe('new task from a lane', () => {
  it.each(LANES)('a %s lane "%s" task lands in that lane, even on a project-scoped board', (by, key) => {
    const task = created(dialogBody(by, key), { boardProject: 'p_board', parentTenant: null })

    expect(laneKey(by, task)).toBe(key)
  })

  it('preselects parked for the unassigned lane and the default row for the default profile', () => {
    expect(initialAssignee(lanePreset('assignee', NO_LANE), DEFAULT_PROFILE)).toBe(PARKED)
    expect(initialAssignee(lanePreset('assignee', DEFAULT_PROFILE), DEFAULT_PROFILE)).toBe('')
    expect(initialAssignee(undefined, DEFAULT_PROFILE)).toBe('')
  })
})
