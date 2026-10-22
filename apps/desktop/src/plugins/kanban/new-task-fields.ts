/** How the new-task dialog turns its assignee choice and a lane preset into
 *  the POST /tasks body. Pure, so the "a task created in a lane lands in that
 *  lane" contract is testable against the backend's create-time defaults. */

import type { LanePreset } from './swimlanes'
import { columnLabel, type KanbanText } from './ui'

/** Assignee select value for "no assignee (parked)". '' means "the default". */
export const PARKED = '__parked__'

/** Initial assignee choice for a dialog opened from a lane: '' (the default
 *  row), PARKED for the unassigned lane, else the lane's profile. Independent
 *  of the resolved default, which may arrive after the dialog opens. */
export function initialAssignee(preset: LanePreset | undefined): string {
  return preset?.assignee === '' ? PARKED : (preset?.assignee ?? '')
}

/** Select value for a choice: the default profile IS the "default" row. */
export const assigneeSelectValue = (choice: string, resolvedDefault: string, defaultRow: string): string =>
  choice && choice !== resolvedDefault ? choice : defaultRow

/** The assignee the dialog submits for a select value (undefined = none). */
export const submittedAssignee = (value: string, resolvedDefault: string): string | undefined =>
  value === PARKED ? undefined : value || resolvedDefault

/** Lane identity the form doesn't edit. An explicit `project_id: ''` stops a
 *  project-scoped board from adopting its project for the "No project" lane;
 *  an absent field keeps the backend's normal defaults. */
export function laneCreateFields(preset: LanePreset | undefined): { project_id?: string; tenant?: string } {
  return {
    ...(preset?.project_id !== undefined && { project_id: preset.project_id }),
    ...(preset?.tenant && { tenant: preset.tenant })
  }
}

/** Assignee rows after the "default" row: the roster minus the default, plus
 *  a lane's assignee that isn't a local profile (it must still be selectable). */
export function assigneeOptions(roster: readonly string[], selected: string, resolvedDefault: string): string[] {
  const extra = selected && selected !== PARKED && !roster.includes(selected) ? [selected] : []

  return [...roster, ...extra].filter(name => name !== resolvedDefault)
}

/** "New task in Ready", or "New task in Ready · Beacon" from a lane cell. */
export function newTaskTitle(k: KanbanText, status: null | string, laneTitle?: string): string {
  if (!status) {
    return k.newTask
  }

  return k.newTaskIn(laneTitle ? `${columnLabel(k, status)} · ${laneTitle}` : columnLabel(k, status))
}
