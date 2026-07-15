import {
  decideStudyPlanProposal,
  getStudyOverview,
  getStudyProfile,
  getStudyProjects,
  getStudyReviewDetail,
  getStudyReviewDue,
  getStudyReviewQueue,
  getStudyReviewStats,
  getStudySchedule,
  getStudySchedules,
  setStudyActiveProject,
  submitStudyReviewAttempt,
  updateStudyProfile,
  updateStudySettings
} from '@/hermes'
import type {
  StudyOverviewResponse,
  StudyPlanProposalDecisionResponse,
  StudyProfile,
  StudyProject,
  StudyProjectsResponse,
  StudyReviewDetail,
  StudyReviewDueResponse,
  StudyReviewQueueResponse,
  StudyReviewStatsResponse,
  StudyReviewSubmission,
  StudyReviewSubmissionResponse,
  StudySchedule,
  StudySchedulesResponse,
  StudySettings
} from '@/types/hermes'

export interface StudyReviewDueParams {
  level?: number
  limit?: number
  subject?: string
}

export interface StudyClient {
  decideProposal(
    projectId: string,
    proposalId: string,
    action: 'accept' | 'reject'
  ): Promise<StudyPlanProposalDecisionResponse>
  getOverview(projectId?: string): Promise<StudyOverviewResponse>
  getProfile(): Promise<StudyProfile>
  getProjects(): Promise<StudyProjectsResponse>
  getReviewDetail(note: string): Promise<StudyReviewDetail>
  getReviewDue(params?: StudyReviewDueParams): Promise<StudyReviewDueResponse>
  getReviewQueue(params?: { limit?: number; state?: string }): Promise<StudyReviewQueueResponse>
  getReviewStats(): Promise<StudyReviewStatsResponse>
  getSchedule(projectId: string, scheduleId: string): Promise<StudySchedule>
  getSchedules(projectId: string): Promise<StudySchedulesResponse>
  selectProject(projectId: string): Promise<{ active_project_id: string; project: StudyProject }>
  submitReview(submission: StudyReviewSubmission): Promise<StudyReviewSubmissionResponse>
  updateProfile(profile: Partial<StudyProfile>): Promise<StudyProfile>
  updateSettings(vaultPath: string): Promise<StudySettings>
}

export const httpStudyClient: StudyClient = {
  decideProposal: (...args) => decideStudyPlanProposal(...args),
  getOverview: projectId => getStudyOverview(projectId),
  getProfile: () => getStudyProfile(),
  getProjects: () => getStudyProjects(),
  getReviewDetail: note => getStudyReviewDetail(note),
  getReviewDue: params => getStudyReviewDue(params),
  getReviewQueue: params => getStudyReviewQueue(params),
  getReviewStats: () => getStudyReviewStats(),
  getSchedule: (projectId, scheduleId) => getStudySchedule(projectId, scheduleId),
  getSchedules: projectId => getStudySchedules(projectId),
  selectProject: projectId => setStudyActiveProject(projectId),
  submitReview: submission => submitStudyReviewAttempt(submission),
  updateProfile: profile => updateStudyProfile(profile),
  updateSettings: vaultPath => updateStudySettings(vaultPath)
}

export interface MemoryStudyClientData {
  overviewByProject?: Record<string, StudyOverviewResponse>
  profile?: StudyProfile
  projects?: StudyProjectsResponse
  reviewDetails?: Record<string, StudyReviewDetail>
  reviewDue?: StudyReviewDueResponse
  reviewQueue?: StudyReviewQueueResponse
  reviewStats?: StudyReviewStatsResponse
  scheduleById?: Record<string, StudySchedule>
  schedulesByProject?: Record<string, StudySchedulesResponse>
  settings?: StudySettings
  submissionResult?: StudyReviewSubmissionResponse
}

function missingMemoryResponse(name: string): never {
  throw new Error(`MemoryStudyClient has no ${name} response`)
}

/** Deterministic adapter for action and state-machine tests; it never touches HTTP. */
export class MemoryStudyClient implements StudyClient {
  readonly submissions: StudyReviewSubmission[] = []

  constructor(readonly data: MemoryStudyClientData = {}) {}

  async decideProposal(
    _projectId: string,
    _proposalId: string,
    _action: 'accept' | 'reject'
  ): Promise<StudyPlanProposalDecisionResponse> {
    return missingMemoryResponse('proposal decision')
  }

  async getOverview(projectId?: string): Promise<StudyOverviewResponse> {
    const selected = projectId ?? this.data.projects?.active_project_id ?? undefined

    return (selected && this.data.overviewByProject?.[selected]) || missingMemoryResponse('overview')
  }

  async getProfile(): Promise<StudyProfile> {
    return (
      this.data.profile ?? {
        daily_review_limit: 20,
        review_level_filter: null,
        subject_filter: null
      }
    )
  }

  async getProjects(): Promise<StudyProjectsResponse> {
    return (
      this.data.projects ?? {
        active_project_id: null,
        configured: false,
        projects: [],
        vault_path: ''
      }
    )
  }

  async getReviewDetail(note: string): Promise<StudyReviewDetail> {
    return this.data.reviewDetails?.[note] ?? missingMemoryResponse(`review detail for ${note}`)
  }

  async getReviewDue(): Promise<StudyReviewDueResponse> {
    return (
      this.data.reviewDue ?? {
        configured: true,
        count: 0,
        date: '1970-01-01',
        due: [],
        subjects: [],
        vault_path: null
      }
    )
  }

  async getReviewQueue(): Promise<StudyReviewQueueResponse> {
    return (
      this.data.reviewQueue ?? {
        new_concepts: [],
        new_concepts_total: 0,
        new_examples: [],
        new_examples_total: 0,
        vault_path: ''
      }
    )
  }

  async getReviewStats(): Promise<StudyReviewStatsResponse> {
    return (
      this.data.reviewStats ?? {
        by_level: {},
        cached: false,
        concept_stats: {},
        due_count: 0,
        progress: 0,
        review_streak: 0,
        reviewed_count: 0,
        spacing_coverage: 0,
        total: 0
      }
    )
  }

  async getSchedule(projectId: string, scheduleId: string): Promise<StudySchedule> {
    const schedule = this.data.scheduleById?.[scheduleId]

    if (!schedule || schedule.project_id !== projectId) {
      return missingMemoryResponse(`schedule ${projectId}/${scheduleId}`)
    }

    return schedule
  }

  async getSchedules(projectId: string): Promise<StudySchedulesResponse> {
    return (
      this.data.schedulesByProject?.[projectId] ?? {
        invalid_schedules: [],
        project_id: projectId,
        schedules: []
      }
    )
  }

  async selectProject(projectId: string): Promise<{ active_project_id: string; project: StudyProject }> {
    const projects = await this.getProjects()
    const project = projects.projects.find(candidate => candidate.project_id === projectId)

    if (!project) {
      return missingMemoryResponse(`project ${projectId}`)
    }

    projects.active_project_id = projectId

    return { active_project_id: projectId, project }
  }

  async submitReview(submission: StudyReviewSubmission): Promise<StudyReviewSubmissionResponse> {
    this.submissions.push(submission)

    return this.data.submissionResult ?? missingMemoryResponse('review submission')
  }

  async updateProfile(profile: Partial<StudyProfile>): Promise<StudyProfile> {
    const updated = { ...(await this.getProfile()), ...profile }
    this.data.profile = updated

    return updated
  }

  async updateSettings(vaultPath: string): Promise<StudySettings> {
    const updated = this.data.settings ?? {
      active_project_id: this.data.projects?.active_project_id ?? null,
      configured: true,
      vault_path: vaultPath
    }

    updated.vault_path = vaultPath
    this.data.settings = updated

    return updated
  }
}
