import { describe, expect, it } from 'vitest'

import { validateStudyProject, validateStudySchedule } from './study-schemas'

function validStudyProject() {
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

function validStudySchedule() {
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

function validLearningProjectV2() {
  return {
    schema_version: 'study_project.v2',
    project_id: 'research-agents',
    title: 'Agent Systems Research',
    domain: 'research',
    timezone: 'Asia/Shanghai',
    phase: 'replication',
    domain_pack: 'research.v1',
    workspace_type: 'hybrid',
    artifact_policy: 'lightweight',
    deadline: '2026-12-01',
    tracks: [{ id: 'methods', label: 'Methods' }],
    objectives: [
      {
        objective_id: 'reproduce-routing-result',
        capability: 'Reproduce and explain one routing result.',
        success_criteria: ['Record the command.', 'Explain one limitation.'],
        evidence_targets: ['execution', 'explanation'],
        source_anchors: [{ kind: 'paper', ref: 'doi:10.0000/example', locator: 'section 4' }]
      }
    ],
    prompt_policy: {
      base_max_chars: 2000,
      intent_max_chars: 2500,
      domain_max_chars: 2000,
      project_summary_max_chars: 1200,
      total_max_chars: 6000,
      updates_apply: 'next_session'
    },
    created_at: '2026-07-13T09:00:00+08:00',
    updated_at: '2026-07-13T09:00:00+08:00'
  }
}

describe('StudyOS schema guards', () => {
  it('accepts the kaoyan project and schedule fixtures', () => {
    const projectResult = validateStudyProject(validStudyProject())
    expect(projectResult.ok).toBe(true)
    if (!projectResult.ok) {
      throw new Error(projectResult.errors.join('\n'))
    }

    const schedule = validStudySchedule()
    const scheduleResult = validateStudySchedule(schedule, projectResult.data)

    expect(scheduleResult.ok).toBe(true)
    if (!scheduleResult.ok) {
      throw new Error(scheduleResult.errors.join('\n'))
    }
    expect(scheduleResult.data).toBe(schedule)
  })

  it('accepts domain-neutral v2 projects without exam placeholders', () => {
    const project = validLearningProjectV2()
    const result = validateStudyProject(project)

    expect(result.ok).toBe(true)
    if (!result.ok) {
      throw new Error(result.errors.join('\n'))
    }
    expect(result.data.schema_version).toBe('study_project.v2')
    expect(result.data.exam_date).toBeUndefined()
    expect(result.data.objectives?.[0]?.objective_id).toBe('reproduce-routing-result')
  })

  it('rejects malformed schedules without throwing', () => {
    const projectResult = validateStudyProject(validStudyProject())
    if (!projectResult.ok) {
      throw new Error(projectResult.errors.join('\n'))
    }
    const schedule = validStudySchedule()
    schedule.events[0].start = '2026-07-01T19:00:00'

    expect(() => validateStudySchedule(schedule, projectResult.data)).not.toThrow()
    const result = validateStudySchedule(schedule, projectResult.data)

    expect(result.ok).toBe(false)
    if (result.ok) {
      throw new Error('Expected invalid schedule')
    }
    expect(result.errors).toContain('events[0].start must include timezone offset')
  })

  it('rejects cross-midnight events when duration does not match', () => {
    const projectResult = validateStudyProject(validStudyProject())
    if (!projectResult.ok) {
      throw new Error(projectResult.errors.join('\n'))
    }
    const schedule = validStudySchedule()
    schedule.events[0].start = '2026-07-01T23:30:00+08:00'
    schedule.events[0].end = '2026-07-02T00:15:00+08:00'
    schedule.events[0].duration_minutes = 120

    const result = validateStudySchedule(schedule, projectResult.data)

    expect(result.ok).toBe(false)
    if (result.ok) {
      throw new Error('Expected invalid schedule')
    }
    expect(result.errors).toContain('events[0].duration_minutes does not match start/end')
  })
})
