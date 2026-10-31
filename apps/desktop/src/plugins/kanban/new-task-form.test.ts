import { describe, expect, it } from 'vitest'

import { buildCreateTaskMetadata, toggleSkill } from './new-task-form'

describe('new task form metadata helpers', () => {
  it('maps the explicit project choice into the create payload', () => {
    expect(
      buildCreateTaskMetadata({
        priority: '2',
        projectId: 'p_123',
        selectedSkills: ['sam-project-orchestration'],
        skillsText: ''
      })
    ).toEqual({
      priority: 2,
      project_id: 'p_123',
      skills: ['sam-project-orchestration']
    })
  })

  it('falls back to comma-separated custom skills and dedupes', () => {
    expect(
      buildCreateTaskMetadata({
        priority: '',
        projectId: '',
        selectedSkills: ['github'],
        skillsText: 'github, wordpress, wordpress'
      })
    ).toEqual({
      priority: 0,
      skills: ['github', 'wordpress']
    })
  })

  it('toggles a skill without duplicates', () => {
    expect(toggleSkill(['github'], 'sam-project-orchestration')).toEqual(['github', 'sam-project-orchestration'])
    expect(toggleSkill(['github', 'sam-project-orchestration'], 'github')).toEqual(['sam-project-orchestration'])
  })
})
