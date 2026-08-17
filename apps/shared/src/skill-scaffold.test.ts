import { describe, expect, it } from 'vitest'

import { skillInvocationText } from './skill-scaffold'

// Byte-identical to what agent/skill_commands.py emits — a desktop/TUI talking
// to an older gateway sees exactly these strings.
const BODY = 'SPIN UP A WORKTREE. Never edit the primary checkout.\n'.repeat(20)

const singleSkill = (instruction?: string) =>
  [
    '[IMPORTANT: The user has invoked the "work" skill, indicating they want you to follow its instructions.',
    'The full skill content is loaded below.]',
    '',
    BODY,
    '',
    '[Skill directory: /Users/x/skills/work]',
    ...(instruction
      ? ['', `The user has provided the following instruction alongside the skill invocation: ${instruction}`]
      : [])
  ].join('\n')

const bundle = (instruction?: string) =>
  [
    '[IMPORTANT: The user has invoked the "/clean /work" stacked skill bundle, loading 2 skills together.]',
    '',
    'Skills loaded: clean, work',
    ...(instruction ? ['', `User instruction: ${instruction}`] : []),
    '',
    '[Loaded as part of the stacked skill invocation "clean".]',
    '',
    BODY
  ].join('\n')

describe('skillInvocationText', () => {
  it('renders a single-skill turn as the invocation, never the body', () => {
    const projected = skillInvocationText(singleSkill('fix the title leak'))

    expect(projected).toBe('/work fix the title leak')
    expect(projected).not.toContain('WORKTREE')
  })

  it('renders a bare invocation as just the command', () => {
    expect(skillInvocationText(singleSkill())).toBe('/work')
  })

  it('renders a bundle turn as the typed keys plus the instruction', () => {
    const projected = skillInvocationText(bundle('ship it'))

    expect(projected).toBe('/clean /work ship it')
    expect(projected).not.toContain('WORKTREE')
  })

  it('collapses newlines in a multi-line instruction so the bubble stays one line', () => {
    expect(skillInvocationText(singleSkill('fix the leak\n\nthen ship'))).toBe('/work fix the leak then ship')
  })

  it('leaves ordinary user prose alone', () => {
    expect(skillInvocationText('just a normal message')).toBeNull()
    expect(skillInvocationText('[IMPORTANT: read the docs]')).toBeNull()
  })
})
