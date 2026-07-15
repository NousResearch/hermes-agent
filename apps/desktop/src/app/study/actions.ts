import {
  $studyConfigured,
  $studyError,
  $studyInvalidSchedules,
  $studyLoadState,
  $studyMessage,
  $studyOverview,
  $studyProjects,
  $studySchedules,
  $studySelectedProjectId,
  $studySelectedSchedule,
  $studySelectedScheduleId,
  $studyVaultPath
} from '@/store/study'
import { $reviewCompletedToday } from '@/store/study-review'
import type {
  StudyOverviewResponse,
  StudyProjectsResponse,
  StudySchedule,
  StudySchedulesResponse,
  StudySettings
} from '@/types/hermes'

import type { StudyClient } from './client'

let requestEpoch = 0

interface ProjectSnapshot {
  overview: StudyOverviewResponse
  schedule: null | StudySchedule
  schedules: StudySchedulesResponse
}

function beginRequest(): number {
  requestEpoch += 1
  $studyLoadState.set('loading')
  $studyError.set(null)

  return requestEpoch
}

function isCurrent(epoch: number): boolean {
  return requestEpoch === epoch
}

function reportError(epoch: number, error: unknown): void {
  if (!isCurrent(epoch)) {
    return
  }

  $studyError.set(error instanceof Error ? error.message : String(error))
  $studyLoadState.set('error')
}

async function fetchProjectSnapshot(client: StudyClient, projectId: string): Promise<ProjectSnapshot> {
  const [schedules, overview] = await Promise.all([client.getSchedules(projectId), client.getOverview(projectId)])
  const scheduleId = schedules.schedules[0]?.schedule_id ?? null
  const schedule = scheduleId ? await client.getSchedule(projectId, scheduleId) : null

  return { overview, schedule, schedules }
}

function commitProjectSnapshot(projectId: string, snapshot: ProjectSnapshot): void {
  $studySelectedProjectId.set(projectId)
  $studySchedules.set(snapshot.schedules.schedules)
  $studyInvalidSchedules.set(snapshot.schedules.invalid_schedules ?? [])
  $studyOverview.set(snapshot.overview)
  $reviewCompletedToday.set(snapshot.overview.completed_today)
  $studySelectedScheduleId.set(snapshot.schedule?.schedule_id ?? null)
  $studySelectedSchedule.set(snapshot.schedule)
}

function clearProjectSnapshot(): void {
  $studySelectedProjectId.set(null)
  $studySchedules.set([])
  $studyInvalidSchedules.set([])
  $studySelectedScheduleId.set(null)
  $studySelectedSchedule.set(null)
  $studyOverview.set(null)
}

function commitProjects(response: StudyProjectsResponse): void {
  $studyConfigured.set(response.configured)
  $studyMessage.set(response.message ?? null)
  $studyProjects.set(response.projects)
  $studyVaultPath.set(response.vault_path ?? null)
}

export async function loadWorkspace(client: StudyClient): Promise<void> {
  const epoch = beginRequest()

  try {
    const projects = await client.getProjects()

    const projectId = projects.projects.some(project => project.project_id === projects.active_project_id)
      ? (projects.active_project_id ?? null)
      : null

    const snapshot = projects.configured && projectId ? await fetchProjectSnapshot(client, projectId) : null

    if (!isCurrent(epoch)) {
      return
    }

    commitProjects(projects)

    if (snapshot && projectId) {
      commitProjectSnapshot(projectId, snapshot)
    } else {
      clearProjectSnapshot()
    }

    $studyLoadState.set('ready')
  } catch (error) {
    reportError(epoch, error)
  }
}

export async function selectProject(client: StudyClient, projectId: string): Promise<void> {
  const epoch = beginRequest()

  try {
    await client.selectProject(projectId)
    const snapshot = await fetchProjectSnapshot(client, projectId)

    if (!isCurrent(epoch)) {
      return
    }

    commitProjectSnapshot(projectId, snapshot)
    $studyLoadState.set('ready')
  } catch (error) {
    reportError(epoch, error)
  }
}

export async function selectSchedule(client: StudyClient, projectId: string, scheduleId: string): Promise<void> {
  const epoch = beginRequest()

  try {
    const schedule = await client.getSchedule(projectId, scheduleId)

    if (!isCurrent(epoch) || $studySelectedProjectId.get() !== projectId) {
      return
    }

    $studySelectedScheduleId.set(scheduleId)
    $studySelectedSchedule.set(schedule)
    $studyLoadState.set('ready')
  } catch (error) {
    reportError(epoch, error)
  }
}

export async function decideProposal(
  client: StudyClient,
  projectId: string,
  proposalId: string,
  action: 'accept' | 'reject'
): Promise<void> {
  const epoch = beginRequest()

  try {
    await client.decideProposal(projectId, proposalId, action)
    const overview = await client.getOverview(projectId)

    if (!isCurrent(epoch) || $studySelectedProjectId.get() !== projectId) {
      return
    }

    $studyOverview.set(overview)
    $reviewCompletedToday.set(overview.completed_today)
    $studyLoadState.set('ready')
  } catch (error) {
    reportError(epoch, error)
  }
}

export async function saveSettings(client: StudyClient, vaultPath: string): Promise<StudySettings> {
  const result = await client.updateSettings(vaultPath)
  await loadWorkspace(client)

  return result
}

export function setStudyMessage(message: null | string): void {
  $studyMessage.set(message)
}

export function cancelStudyActions(): void {
  requestEpoch += 1
}
