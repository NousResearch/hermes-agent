/** How the new-task dialog turns its assignee choice and a lane preset into
 *  the POST /tasks body. Pure, so the "a task created in a lane lands in that
 *  lane" contract is testable against the backend's create-time defaults. */

import type { LanePreset } from './swimlanes'

/** Assignee select value for "no assignee (parked)". '' means "the default". */
export const PARKED = '__parked__'

/** Initial assignee select value for a dialog opened from a lane. */
export function initialAssignee(preset: LanePreset | undefined, resolvedDefault: string): string {
  if (preset?.assignee === undefined) {
    return ''
  }

  // The unassigned lane asks for no assignee; the default profile is already
  // the "default" row, so it maps there rather than to a duplicate entry.
  return preset.assignee === '' ? PARKED : preset.assignee === resolvedDefault ? '' : preset.assignee
}

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
