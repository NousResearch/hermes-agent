import { atom } from 'nanostores'

import type {
  InvalidStudySchedule,
  StudyOverviewResponse,
  StudyProject,
  StudySchedule,
  StudyScheduleSummary
} from '@/types/hermes'

export type StudyLoadState = 'idle' | 'loading' | 'ready' | 'error'

export const $studyProjects = atom<StudyProject[]>([])
export const $studySelectedProjectId = atom<null | string>(null)
export const $studySchedules = atom<StudyScheduleSummary[]>([])
export const $studyInvalidSchedules = atom<InvalidStudySchedule[]>([])
export const $studySelectedScheduleId = atom<null | string>(null)
export const $studySelectedSchedule = atom<null | StudySchedule>(null)
export const $studyOverview = atom<null | StudyOverviewResponse>(null)
export const $studyVaultPath = atom<null | string>(null)
export const $studyLoadState = atom<StudyLoadState>('idle')
export const $studyError = atom<null | string>(null)
export const $studyConfigured = atom(true)
export const $studyMessage = atom<null | string>(null)

export function resetStudyState(): void {
  $studyProjects.set([])
  $studySelectedProjectId.set(null)
  $studySchedules.set([])
  $studyInvalidSchedules.set([])
  $studySelectedScheduleId.set(null)
  $studySelectedSchedule.set(null)
  $studyOverview.set(null)
  $studyVaultPath.set(null)
  $studyLoadState.set('idle')
  $studyError.set(null)
  $studyConfigured.set(true)
  $studyMessage.set(null)
}
