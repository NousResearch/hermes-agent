import { afterEach, describe, expect, it } from 'vitest'

import {
  $studyLoadState,
  $studyOverview,
  $studyProjects,
  $studySelectedProjectId,
  resetStudyState
} from '@/store/study'
import type { StudyOverviewResponse, StudyProject } from '@/types/hermes'

import { cancelStudyActions, loadWorkspace, selectProject } from './actions'
import { MemoryStudyClient } from './client'

function project(projectId: string): StudyProject {
  return {
    artifact_policy: 'lightweight',
    created_at: '2026-07-15T00:00:00Z',
    domain: 'research',
    domain_pack: 'research.v1',
    objectives: [],
    phase: 'active',
    project_id: projectId,
    prompt_policy: {
      base_max_chars: 2000,
      domain_max_chars: 2000,
      intent_max_chars: 2000,
      project_summary_max_chars: 1200,
      total_max_chars: 6000,
      updates_apply: 'next_session'
    },
    schema_version: 'study_project.v2',
    timezone: 'Asia/Shanghai',
    title: projectId,
    tracks: [],
    updated_at: '2026-07-15T00:00:00Z',
    workspace_type: 'hybrid'
  }
}

function overview(value: StudyProject, completedToday = 0): StudyOverviewResponse {
  return {
    active_project_id: value.project_id,
    activity_today: 0,
    as_of: '2026-07-15T00:00:00Z',
    completed_today: completedToday,
    configured: true,
    due_reviews: { count: 0, items: [], scope: 'vault', subjects: [] },
    evidence: {
      assistance_provenance: {},
      attempt_count: 0,
      dimensions: {},
      evaluator_provenance: {},
      independently_verified_count: 0,
      latest_evidence_at: null
    },
    intervention_queue: {
      as_of: '2026-07-15T00:00:00Z',
      generated_at: '2026-07-15T00:00:00Z',
      items: [],
      project_id: value.project_id,
      warnings: []
    },
    pending_plan_proposals: [],
    project: value,
    today: '2026-07-15',
    today_events: [],
    vault_path: '/tmp/study'
  }
}

function memoryClient(projects: StudyProject[]): MemoryStudyClient {
  return new MemoryStudyClient({
    overviewByProject: Object.fromEntries(projects.map(value => [value.project_id, overview(value)])),
    projects: {
      active_project_id: projects[0]?.project_id ?? null,
      configured: true,
      projects,
      vault_path: '/tmp/study'
    }
  })
}

afterEach(() => {
  cancelStudyActions()
  resetStudyState()
})

describe('StudyOS desktop actions', () => {
  it('loads an atomic workspace snapshot through the in-memory adapter', async () => {
    const selected = project('research-agents')
    await loadWorkspace(memoryClient([selected]))

    expect($studyLoadState.get()).toBe('ready')
    expect($studyProjects.get()).toEqual([selected])
    expect($studySelectedProjectId.get()).toBe(selected.project_id)
    expect($studyOverview.get()?.project.project_id).toBe(selected.project_id)
  })

  it('does not let an older project response overwrite a newer selection', async () => {
    const slow = project('slow-project')
    const fast = project('fast-project')
    let releaseSlow: (() => void) | undefined

    const slowGate = new Promise<void>(resolve => {
      releaseSlow = resolve
    })

    const client = memoryClient([slow, fast])
    const getOverview = client.getOverview.bind(client)

    client.getOverview = async projectId => {
      if (projectId === slow.project_id) {
        await slowGate
      }

      return getOverview(projectId)
    }

    const oldSelection = selectProject(client, slow.project_id)
    await Promise.resolve()
    await selectProject(client, fast.project_id)
    releaseSlow?.()
    await oldSelection

    expect($studySelectedProjectId.get()).toBe(fast.project_id)
    expect($studyOverview.get()?.project.project_id).toBe(fast.project_id)
    expect($studyLoadState.get()).toBe('ready')
  })
})
