import { atom } from 'nanostores'

import type { StudyProject, StudySchedule, StudyScheduleSummary } from '@/types/hermes'

export type StudyLoadState = 'idle' | 'loading' | 'ready' | 'error'

export const $studyProjects = atom<StudyProject[]>([])
export const $studySelectedProjectId = atom<null | string>(null)
export const $studySchedules = atom<StudyScheduleSummary[]>([])
export const $studySelectedScheduleId = atom<null | string>(null)
export const $studySelectedSchedule = atom<null | StudySchedule>(null)
export const $studyLoadState = atom<StudyLoadState>('idle')
export const $studyError = atom<null | string>(null)
export const $studyConfigured = atom(true)
export const $studyMessage = atom<null | string>(null)

export function resetStudyState(): void {
  $studyProjects.set([])
  $studySelectedProjectId.set(null)
  $studySchedules.set([])
  $studySelectedScheduleId.set(null)
  $studySelectedSchedule.set(null)
  $studyLoadState.set('idle')
  $studyError.set(null)
  $studyConfigured.set(true)
  $studyMessage.set(null)
}
