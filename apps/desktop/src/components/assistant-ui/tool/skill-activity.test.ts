import { describe, expect, it } from 'vitest'

import { buildToolView } from './fallback-model'
import { summarizeToolRun } from './run-summary'

const skill = { type: 'tool-call' as const, toolName: 'skill_view', args: { name: 'research-notes' } }

describe('skill activity identity', () => {
  it('names instruction loads, resource reads, failures and unobserved outcomes without claiming task completion', () => {
    expect(buildToolView(skill, '').title).toBe('Loading skill: research-notes')
    expect(buildToolView({ ...skill, result: { success: true } }, '').title).toBe('Loaded skill: research-notes')
    const resource = { ...skill, args: { ...skill.args, file_path: 'references/example.md' } }
    expect(buildToolView(resource, '').title).toBe('Reading skill resource: research-notes → references/example.md')
    expect(buildToolView({ ...resource, result: 'resource body' }, '').title).toBe(
      'Read skill resource: research-notes → references/example.md'
    )
    expect(buildToolView({ ...skill, isError: true, result: 'missing' }, '').title).toBe(
      'Failed to load skill: research-notes'
    )
    expect(summarizeToolRun([skill], false)).toBe('Skill result unavailable: research-notes')
  })

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
