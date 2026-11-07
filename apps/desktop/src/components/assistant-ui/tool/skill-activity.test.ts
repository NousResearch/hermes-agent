import { describe, expect, it } from 'vitest'

import { summarizeToolRun } from './run-summary'
import { skillActivityTitle } from './skill-activity'

const skill = { type: 'tool-call' as const, toolName: 'skill_view', args: { name: 'research-notes' } }

/** The stub `tools/skills_tool_dedup.py` serves for a repeat view of a skill file. */
const unchangedStub = (file = 'SKILL.md') => ({
  success: true,
  status: 'unchanged',
  name: 'research-notes',
  file,
  dedup: true,
  content_returned: false,
  message: 'Skill content unchanged since it was loaded earlier in this conversation — refer to the earlier result.'
})

describe('skill activity identity', () => {
  it('keeps each skill identity and failed outcome in mixed collapsed summaries', () => {
    const summary = summarizeToolRun(
      [
        { ...skill, result: 'instructions' },
        { ...skill, args: { name: 'second-skill' }, result: { error: 'not found' } },
        { toolName: 'terminal', args: { command: 'echo ok' }, result: { stdout: 'ok' } }
      ],
      false
    )

    expect(summary).toContain('Loaded skill: research-notes')
    expect(summary).toContain('failed to load skill: second-skill')
    expect(summary).toContain('ran 1 command')
    expect(summary).toContain('1 tool call failed')
  })
})

describe('skill activity for a deduplicated skill_view', () => {
  it('labels the unchanged stub as unchanged, not as another load', () => {
    expect(skillActivityTitle({ ...skill, result: JSON.stringify(unchangedStub()) })).toBe(
      'Skill unchanged: research-notes'
    )
  })

  it('reads a restored-history object result the same way as the live string result', () => {
    expect(skillActivityTitle({ ...skill, result: unchangedStub() })).toBe('Skill unchanged: research-notes')
  })

  it('keeps the skill resource in the target of an unchanged resource read', () => {
    expect(
      skillActivityTitle({
        ...skill,
        args: { name: 'research-notes', file_path: 'references/api.md' },
        result: unchangedStub('references/api.md')
      })
    ).toBe('Skill unchanged: research-notes → references/api.md')
  })

  it('keeps a real load, a pending call, a failure and a missing result distinct', () => {
    expect(skillActivityTitle({ ...skill, result: { success: true, name: 'research-notes', content: 'body' } })).toBe(
      'Loaded skill: research-notes'
    )
    expect(skillActivityTitle(skill)).toBe('Loading skill: research-notes')
    expect(skillActivityTitle({ ...skill, completedAt: 1, isError: true, result: { success: false } })).toBe(
      'Failed to load skill: research-notes'
    )
    expect(skillActivityTitle({ ...skill, completedAt: 1 }, false)).toBe('Skill result unavailable: research-notes')
  })

  it('narrates the unchanged repeat beside a load in the collapsed run summary', () => {
    const summary = summarizeToolRun(
      [
        { ...skill, result: 'instructions' },
        { ...skill, result: JSON.stringify(unchangedStub()) }
      ],
      false
    )

    expect(summary).toContain('Loaded skill: research-notes')
    expect(summary).toContain('skill unchanged: research-notes')
  })
})
