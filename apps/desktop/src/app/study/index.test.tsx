import { cleanup, render, screen, waitFor } from '@testing-library/react'
import { I18nProvider } from '@/i18n/context'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const getStudyProjects = vi.fn()
const getStudySchedules = vi.fn()
const getStudySchedule = vi.fn()

vi.mock('@/hermes', () => ({
  getStudyProjects: () => getStudyProjects(),
  getStudySchedules: (projectId: string) => getStudySchedules(projectId),
  getStudySchedule: (projectId: string, scheduleId: string) => getStudySchedule(projectId, scheduleId)
}))

function project() {
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

function schedule() {
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

function learningProject() {
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

function researchSchedule() {
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
  getStudyProjects.mockResolvedValue({ configured: true, projects: [p], vault_path: '/tmp/vault' })
  getStudySchedules.mockResolvedValue({
    project_id: p.project_id,
    schedules: [
      {
        schedule_id: s.schedule_id,
        project_id: s.project_id,
        title: s.title,
        timezone: s.timezone,
        range: s.range,
        event_count: s.events.length
      }
    ]
  })
  getStudySchedule.mockResolvedValue(s)
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

describe('StudyView', () => {
  it('renders valid StudyOS schedule entries', async () => {
    await renderStudy()

    expect(await screen.findAllByText('2027 考研学习计划')).not.toHaveLength(0)
    expect(await screen.findByText('数学：导数定义整理')).toBeTruthy()
    expect(screen.getByText(/2026-07-01 19:00/)).toBeTruthy()
    expect(screen.queryByRole('button', { name: /delete|edit|create/i })).toBeNull()
  })

  it('renders an unconfigured vault response without throwing', async () => {
    getStudyProjects.mockResolvedValue({ configured: false, projects: [], message: 'StudyOS vault not configured' })

    await renderStudy()

    expect(await screen.findByText('StudyOS vault not configured')).toBeTruthy()
    expect(getStudySchedules).not.toHaveBeenCalled()
  })

  it('does not render events from an invalid schedule', async () => {
    const invalid = schedule()
    invalid.events[0].start = '2026-07-01T19:00:00'
    getStudySchedule.mockResolvedValue(invalid)

    await renderStudy()

    expect(await screen.findByText('This StudyOS schedule is invalid and was not rendered.')).toBeTruthy()
    await waitFor(() => expect(screen.queryByText('数学：导数定义整理')).toBeNull())
  })

  it('renders domain-neutral projects with tracks and a deadline', async () => {
    const p = learningProject()
    const s = researchSchedule()
    getStudyProjects.mockResolvedValue({ configured: true, projects: [p], vault_path: '/tmp/vault' })
    getStudySchedules.mockResolvedValue({
      project_id: p.project_id,
      schedules: [
        {
          schedule_id: s.schedule_id,
          project_id: s.project_id,
          title: s.title,
          timezone: s.timezone,
          range: s.range,
          event_count: s.events.length
        }
      ]
    })
    getStudySchedule.mockResolvedValue(s)

    await renderStudy()

    expect(await screen.findAllByText('Agent Systems Research')).not.toHaveLength(0)
    expect(screen.getByText('Deadline')).toBeTruthy()
    expect(screen.getAllByText('2026-12-01')).not.toHaveLength(0)
    expect(await screen.findByText('Methods：Replicate routing result')).toBeTruthy()
    expect(screen.queryByText(/undefined/)).toBeNull()
  })
})
