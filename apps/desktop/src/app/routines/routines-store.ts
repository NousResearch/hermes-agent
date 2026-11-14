import { atom } from 'nanostores'

import type { CronJob } from '@/hermes'

/** A routine is (agent, id): two agents can each own a `daily-brief`. */
export interface RoutineRef {
  id: string
  profile: string
}

export const routineKey = (job: Pick<CronJob, 'id' | 'profile'>) => `${job.profile ?? ''}\u0000${job.id}`

export const sameRoutine = (job: Pick<CronJob, 'id' | 'profile'>, ref: null | RoutineRef) =>
  Boolean(ref) && job.id === ref?.id && (job.profile ?? '') === ref.profile

/** What the rail page's stage shows: the overview, or one routine. In memory only, so every window
 *  keeps its own and reopening Routines returns to where this window left it (as Agents does). */
export const $routineSelection = atom<null | RoutineRef>(null)

export function selectRoutine(ref: null | RoutineRef): void {
  $routineSelection.set(ref)
}

/** The one editor (R3 → R4). `pick` is "What kind of routine?"; `create` is the form, blank or seeded
 *  from a ready-made routine; `edit` changes an existing one. `lockProfile` keeps the agent fixed
 *  (an agent's own Settings › Routines). */
export type RoutineEditor =
  | { mode: 'closed' }
  | { mode: 'create'; blueprintKey?: string; fromPick: boolean; lockProfile: boolean; profile: string }
  | { mode: 'edit'; job: CronJob }
  | { mode: 'pick'; lockProfile: boolean; profile: string }

export const $routineEditor = atom<RoutineEditor>({ mode: 'closed' })

export function openRoutineEditor(editor: RoutineEditor): void {
  $routineEditor.set(editor)
}

export function closeRoutineEditor(): void {
  $routineEditor.set({ mode: 'closed' })
}

/** The routine a Delete is waiting to be confirmed for. */
export const $routinePendingDelete = atom<CronJob | null>(null)

/** The routine a Rename (a rail quick action) is open for. */
export const $routineRename = atom<CronJob | null>(null)

/** A search the Routines overview opens with (the chat sidebar's per-agent chip names its agent), taken
 *  once so reopening Routines later starts unfiltered. */
export const $routineOverviewQuery = atom('')

export function takeRoutineOverviewQuery(): string {
  const query = $routineOverviewQuery.get()
  $routineOverviewQuery.set('')

  return query
}
