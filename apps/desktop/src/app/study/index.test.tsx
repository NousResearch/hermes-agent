import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n/context'
import type { StudyOverviewResponse, StudyProject, StudySchedule } from '@/types/hermes'

const getStudyProjects = vi.fn()
const getStudySchedules = vi.fn()
const getStudySchedule = vi.fn()
const getStudyOverview = vi.fn()
const setStudyActiveProject = vi.fn()
const updateStudySettings = vi.fn()
const decideStudyPlanProposal = vi.fn()

vi.mock('@/hermes', () => ({
  getStudyProjects: () => getStudyProjects(),
  getStudySchedules: (projectId: string) => getStudySchedules(projectId),
  getStudySchedule: (projectId: string, scheduleId: string) => getStudySchedule(projectId, scheduleId),
  getStudyOverview: (projectId: string) => getStudyOverview(projectId),
  setStudyActiveProject: (projectId: string) => setStudyActiveProject(projectId),
  updateStudySettings: (path: string) => updateStudySettings(path),
  decideStudyPlanProposal: (projectId: string, proposalId: string, action: string) =>
    decideStudyPlanProposal(projectId, proposalId, action)
}))

function project(): StudyProject {
  return {
    schema_version: 'study_project.v1',
    project_id: 'kaoyan-2027',
    title: '2027 考研学习计划',
    domain: 'kaoyan',
    exam_type: '考研',
    exam_date: '2027-12-20',
    timezone: 'Asia/Shanghai',
    phase: 'foundation',
    domain_pack: 'kaoyan.v1',
    subjects: [
      { id: 'math', label: '数学', target_score: 120 },
      { id: 'english', label: '英语一', target_score: 75 },
      { id: 'politics', label: '政治', target_score: 75 }
    ],
    prompt_policy: {
      base_max_chars: 2000,
      intent_max_chars: 2500,
      domain_max_chars: 2000,
      project_summary_max_chars: 1200,
      total_max_chars: 6000,
      updates_apply: 'next_session'
    },
    created_at: '2026-06-28T00:00:00+08:00',
    updated_at: '2026-06-28T00:00:00+08:00'
  }
}

function schedule(): StudySchedule {
  return {
    schema_version: 'study_schedule.v1',
    schedule_id: 'kaoyan-2027-master-plan',
    project_id: 'kaoyan-2027',
    title: '2027 考研数学基础阶段计划',
    timezone: 'Asia/Shanghai',
    range: { start: '2026-07-01', end: '2026-07-31' },
    phases: [
      {
        id: 'foundation',
        title: '基础阶段',
        start: '2026-07-01',
        end: '2026-09-30',
        goal: '完成核心考点覆盖'
      }
    ],
    events: [
      {
        id: 'evt-20260701-math-derivative',
        title: '数学：导数定义整理',
        subject_id: 'math',
        type: 'learning',
        start: '2026-07-01T19:00:00+08:00',
        end: '2026-07-01T21:00:00+08:00',
        duration_minutes: 120,
        goals: ['整理导数定义例题'],
        source_curriculum: '一元函数微分学',
        status: 'planned'
      }
    ]
  }
}

function learningProject(): StudyProject {
  return {
    ...project(),
    schema_version: 'study_project.v2',
    project_id: 'research-agents',
    title: 'Agent Systems Research',
    domain: 'research',
    domain_pack: 'research.v1',
    phase: 'replication',
    deadline: '2026-12-01',
    exam_type: undefined,
    exam_date: undefined,
    subjects: undefined,
    workspace_type: 'hybrid',
    artifact_policy: 'lightweight',
    tracks: [{ id: 'methods', label: 'Methods' }],
    objectives: [
      {
        objective_id: 'reproduce-routing-result',
        capability: 'Reproduce and explain one routing result.',
        success_criteria: ['Record the command.', 'Explain one limitation.'],
        evidence_targets: ['execution', 'explanation']
      }
    ]
  }
}

function researchSchedule(): StudySchedule {
  const base = schedule()
  return {
    ...base,
    schedule_id: 'research-agents-master-plan',
    project_id: 'research-agents',
    title: 'Agent Systems Research Plan',
    events: [
      {
        ...base.events[0],
        id: 'evt-20260701-routing-replication',
        title: 'Replicate routing result',
        subject_id: 'methods'
      }
    ]
  }
}

function overview(p: StudyProject = project(), s: StudySchedule = schedule()): StudyOverviewResponse {
  return {
    configured: true,
    vault_path: '/tmp/vault',
    active_project_id: p.project_id,
    project: p,
    as_of: '2026-07-01T12:00:00+08:00',
    today: '2026-07-01',
    today_events: s.events.map(event => ({ ...event, schedule_id: s.schedule_id, schedule_title: s.title })),
    due_reviews: { scope: 'vault', count: 0, subjects: [], items: [] },
    completed_today: 0,
    activity_today: 0,
    evidence: {
      attempt_count: 0,
      independently_verified_count: 0,
      latest_evidence_at: null,
      dimensions: {},
      evaluator_provenance: {},
      assistance_provenance: {}
    },
    intervention_queue: {
      project_id: p.project_id,
      generated_at: '2026-07-01T12:00:00+08:00',
      as_of: '2026-07-01T12:00:00+08:00',
      items: [],
      warnings: []
    },
    pending_plan_proposals: []
  }
}

function renderStudy() {
  return import('./index').then(({ StudyView }) =>
    render(
      <I18nProvider configClient={null}>
        <StudyView />
      </I18nProvider>
    )
  )
}

beforeEach(() => {
  const p = project()
  const s = schedule()
  getStudyProjects.mockResolvedValue({
    active_project_id: p.project_id,
    configured: true,
    projects: [p],
    vault_path: '/tmp/vault'
  })
  getStudySchedules.mockResolvedValue({
    project_id: p.project_id,
    schedules: [
      {
        schedule_id: s.schedule_id,
        project_id: s.project_id,
        title: s.title,
        timezone: s.timezone,
        range: s.range,
        phase_count: s.phases.length,
        event_count: s.events.length
      }
    ],
    invalid_schedules: []
  })
  getStudySchedule.mockResolvedValue(s)
  getStudyOverview.mockResolvedValue(overview(p, s))
  setStudyActiveProject.mockResolvedValue({ active_project_id: p.project_id, project: p })
  updateStudySettings.mockResolvedValue({ configured: true, vault_path: '/tmp/vault' })
  decideStudyPlanProposal.mockResolvedValue({ changed: true, schedule_mutated: false })
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

describe('StudyView', () => {
  it('renders valid StudyOS schedule entries', async () => {
    await renderStudy()

    expect(await screen.findAllByText('2027 考研学习计划')).not.toHaveLength(0)
    fireEvent.click(await screen.findByRole('button', { name: 'Calendar' }))
    expect(await screen.findByText('数学：导数定义整理')).toBeTruthy()
    expect(screen.getByText(/2026-07-01 19:00/)).toBeTruthy()
    expect(screen.queryByRole('button', { name: /delete|edit|create/i })).toBeNull()
  })

  it('renders long-term phases without inventing daily schedule events', async () => {
    const roadmap = schedule()
    roadmap.title = '暑期长期路线图'
    roadmap.range = { start: '2026-07-16', end: '2026-10-21' }
    roadmap.phases = [
      {
        id: 'topic-coverage',
        title: '专题覆盖',
        start: '2026-07-16',
        end: '2026-08-31',
        goal: '完成核心专题覆盖',
        effort_minutes: 3600,
        goals: ['上午完成专题', '下午完成概率'],
        source_curricula: ['空间解析几何', '概率'],
        status: 'planned'
      }
    ]
    roadmap.events = []
    getStudySchedules.mockResolvedValue({
      project_id: roadmap.project_id,
      schedules: [
        {
          schedule_id: roadmap.schedule_id,
          project_id: roadmap.project_id,
          title: roadmap.title,
          timezone: roadmap.timezone,
          range: roadmap.range,
          phase_count: roadmap.phases.length,
          event_count: 0
        }
      ],
      invalid_schedules: []
    })
    getStudySchedule.mockResolvedValue(roadmap)
    getStudyOverview.mockResolvedValue(overview(project(), roadmap))

    await renderStudy()
    fireEvent.click(await screen.findByRole('button', { name: 'Calendar' }))

    expect(await screen.findByText('专题覆盖')).toBeTruthy()
    expect(screen.getByText('完成核心专题覆盖')).toBeTruthy()
    expect(screen.getByText('60h')).toBeTruthy()
    expect(screen.getByText('上午完成专题')).toBeTruthy()
    expect(screen.getByText('No concrete study sessions scheduled.')).toBeTruthy()
  })

  it('surfaces invalid schedule files returned by discovery', async () => {
    getStudySchedules.mockResolvedValue({
      project_id: 'kaoyan-2027',
      schedules: [],
      invalid_schedules: [
        {
          schedule_id: 'kaoyan-math-2026-summer-v3',
          path: '.StudyOS/projects/kaoyan-2027/schedules/kaoyan-math-2026-summer-v3.json',
          errors: ['events[0].duration_minutes must be an integer from 1 to 720']
        }
      ]
    })

    await renderStudy()

    expect(await screen.findByText('kaoyan-math-2026-summer-v3')).toBeTruthy()
    expect(screen.getByText('events[0].duration_minutes must be an integer from 1 to 720')).toBeTruthy()
    expect(screen.queryByText('No schedules saved for this project.')).toBeNull()
  })

  it('renders an unconfigured vault response without throwing', async () => {
    getStudyProjects.mockResolvedValue({ configured: false, projects: [], message: 'StudyOS vault not configured' })

    await renderStudy()

    expect(await screen.findByText('Connect your StudyOS Vault')).toBeTruthy()
    expect(getStudySchedules).not.toHaveBeenCalled()
  })

  it('does not render events from an invalid schedule', async () => {
    const invalid = schedule()
    invalid.events[0].start = '2026-07-01T19:00:00'
    getStudySchedule.mockResolvedValue(invalid)

    await renderStudy()

    fireEvent.click(await screen.findByRole('button', { name: 'Calendar' }))

    expect(await screen.findByText('This StudyOS schedule is invalid and was not rendered.')).toBeTruthy()
    await waitFor(() => expect(screen.queryByText('数学：导数定义整理')).toBeNull())
  })

  it('renders domain-neutral projects with tracks and a deadline', async () => {
    const p = learningProject()
    const s = researchSchedule()
    getStudyProjects.mockResolvedValue({
      active_project_id: p.project_id,
      configured: true,
      projects: [p],
      vault_path: '/tmp/vault'
    })
    getStudySchedules.mockResolvedValue({
      project_id: p.project_id,
      schedules: [
        {
          schedule_id: s.schedule_id,
          project_id: s.project_id,
          title: s.title,
          timezone: s.timezone,
          range: s.range,
          phase_count: s.phases.length,
          event_count: s.events.length
        }
      ],
      invalid_schedules: []
    })
    getStudySchedule.mockResolvedValue(s)
    getStudyOverview.mockResolvedValue(overview(p, s))

    await renderStudy()

    fireEvent.click(await screen.findByRole('button', { name: 'Calendar' }))

    expect(await screen.findAllByText('Agent Systems Research')).not.toHaveLength(0)
    expect(screen.getByText('Deadline')).toBeTruthy()
    expect(screen.getAllByText('2026-12-01')).not.toHaveLength(0)
    expect(await screen.findByText('Methods：Replicate routing result')).toBeTruthy()
    expect(screen.queryByText(/undefined/)).toBeNull()
  })

  it('loads the shared active project and persists an explicit switch', async () => {
    const first = project()
    const active = learningProject()
    const activeSchedule = researchSchedule()
    getStudyProjects.mockResolvedValue({
      active_project_id: active.project_id,
      configured: true,
      projects: [first, active],
      vault_path: '/tmp/vault'
    })
    getStudySchedules.mockResolvedValue({ project_id: active.project_id, schedules: [] })
    getStudyOverview.mockResolvedValue(overview(active, activeSchedule))

    await renderStudy()

    await waitFor(() => expect(getStudySchedules).toHaveBeenCalledWith('research-agents'))
    fireEvent.click(screen.getByRole('button', { name: /2027 考研学习计划/ }))
    await waitFor(() => expect(setStudyActiveProject).toHaveBeenCalledWith('kaoyan-2027'))
  })

  it('saves a first-time Vault setting through the setup surface', async () => {
    const p = project()
    getStudyProjects
      .mockResolvedValueOnce({ configured: false, projects: [], message: 'StudyOS vault not configured' })
      .mockResolvedValueOnce({
        active_project_id: p.project_id,
        configured: true,
        projects: [p],
        vault_path: '/study/vault'
      })

    await renderStudy()

    fireEvent.change(await screen.findByLabelText('Vault path'), { target: { value: '/study/vault' } })
    fireEvent.click(screen.getByRole('button', { name: 'Save and enable StudyOS' }))
    await waitFor(() => expect(updateStudySettings).toHaveBeenCalledWith('/study/vault'))
  })

  it('requires an explicit inbox decision and delegates it to the backend', async () => {
    const initial = overview()
    const intervention = {
      intervention_id: 'iv-execution',
      objective_id: 'project-readiness',
      capability: 'Solve one transfer problem independently.',
      kind: 'independence_probe',
      evidence_dimension: 'execution',
      priority_score: 80,
      priority_band: 'high' as const,
      reasons: ['Successful execution evidence is not independently verified.'],
      evidence_attempt_ids: ['att-self'],
      recommended_activity: {
        activity_kind: 'independence_probe',
        evidence_target: 'execution',
        assistance_level: 'independent',
        duration_minutes: 30,
        requires_evaluator: true,
        success_criteria: ['Solve without hints.']
      }
    }
    initial.intervention_queue.items = [intervention]
    initial.pending_plan_proposals = [
      {
        proposal_id: 'plan-execution',
        project_id: 'kaoyan-2027',
        title: 'Independent execution check',
        status: 'proposed',
        rationale: 'Self-reported success needs independent verification.',
        created_at: '2026-07-01T12:00:00+08:00',
        items: [intervention],
        schedule_change: { state: 'not_applied', requires_explicit_save: true }
      }
    ]
    getStudyOverview.mockResolvedValueOnce(initial).mockResolvedValueOnce({
      ...initial,
      pending_plan_proposals: []
    })

    await renderStudy()

    fireEvent.click(await screen.findByRole('button', { name: 'Suggestions' }))
    expect(await screen.findByText('Independent execution check')).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'Accept' }))
    await waitFor(() => expect(decideStudyPlanProposal).toHaveBeenCalledWith('kaoyan-2027', 'plan-execution', 'accept'))
  })
})
