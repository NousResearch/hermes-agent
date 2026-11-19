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

export const refOf = (job: Pick<CronJob, 'id' | 'profile'>): RoutineRef => ({ id: job.id, profile: job.profile ?? '' })

/** The routine the page's stage shows; null shows the first one (the list itself below wide). In
 *  memory only, so every window keeps its own and reopening Routines returns to where it left off. */
export const $routineSelection = atom<null | RoutineRef>(null)

export function selectRoutine(ref: null | RoutineRef): void {
  $routineSelection.set(ref)
}

/** The create / edit sheet: the Routines page is the only place a routine is created or changed.
 *  `profile` preselects the agent of a new routine (an agent's page sends its own). */
export type RoutineSheet =
  | { mode: 'closed' }
  | { mode: 'create'; profile: null | string }
  | { mode: 'edit'; ref: RoutineRef }

export const $routineSheet = atom<RoutineSheet>({ mode: 'closed' })

export function openRoutineSheet(sheet: RoutineSheet): void {
  $routineSheet.set(sheet)
}

export function closeRoutineSheet(): void {
  $routineSheet.set({ mode: 'closed' })
}
