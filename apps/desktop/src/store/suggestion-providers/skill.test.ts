import { describe, expect, it } from 'vitest'

import type { ChatMessage } from '@/lib/chat-messages'

import { collidesWithWorkspace, skillHit, skillPattern, skillTouchedInMessages } from './skill'

// skillHit is the provider's real predicate: a whole-word match that the user
// has finished typing (at least one character follows it).
const hits = (name: string, draft: string) => skillHit(skillPattern(name), draft.toLowerCase())

describe('skillPattern + skillHit', () => {
  it('matches the exact name as a completed whole word', () => {
    expect(hits('perf', 'run the perf loop')).toBe(true)
    expect(hits('perf', 'performance is bad')).toBe(false)
  })

  it('hyphenated names also match spaced phrasing', () => {
    expect(hits('pr-ready', 'make this pr ready pls')).toBe(true)
    expect(hits('pr-ready', 'run pr-ready on it')).toBe(true)
    expect(hits('pr_ready', 'pr ready please')).toBe(true)
  })

  it('never matches inside other words', () => {
    expect(hits('read', 'i already did')).toBe(false)
    expect(hits('work', 'reworked the layout')).toBe(false)
    // Suffix boundary includes hyphen: "clean" must not fire inside "clean-up"
    // (a different skill may own that name).
    expect(hits('clean', 'do a clean-up pass')).toBe(false)
  })

  it('is case-insensitive via lowercased input', () => {
    expect(hits('Clean', 'please clean this diff')).toBe(true)
  })

  it('a name still under the caret is not a hit yet', () => {
    // The debounce fires while the word is the last thing typed — wait for
    // the next keystroke to call it intent.
    expect(hits('perf', 'run perf')).toBe(false)
    expect(hits('perf', 'run perf ')).toBe(true)
    expect(hits('perf', 'run perf.')).toBe(true)
  })

  it('an earlier completed occurrence still counts', () => {
    expect(hits('perf', 'perf first, then more perf')).toBe(true)
  })
})

describe('collidesWithWorkspace', () => {
  it('suppresses a skill named exactly like the cwd folder', () => {
    expect(collidesWithWorkspace('hermes-agent', '/Users/b/www/hermes-agent')).toBe(true)
  })

  it('suppresses inside worktree-suffixed folders too', () => {
    expect(collidesWithWorkspace('hermes-agent', '/Users/b/www/hermes-agent-suggest')).toBe(true)
  })

  it('does not suppress on substring-only overlap', () => {
    // "perf" inside "perfect-app" is not a homonym of the project.
    expect(collidesWithWorkspace('perf', '/Users/b/www/perfect-app')).toBe(false)
    expect(collidesWithWorkspace('clean', '/Users/b/www/hermes-agent')).toBe(false)
  })

  it('never collides when detached (empty cwd)', () => {
    expect(collidesWithWorkspace('hermes-agent', '')).toBe(false)
  })
})

// -- skillTouchedInMessages ---------------------------------------------------

const toolCall = (toolName: string, args?: unknown, argsText = ''): ChatMessage => ({
  id: 't',
  role: 'assistant',
  parts: [{ args: args as never, argsText, toolCallId: 'x', toolName, type: 'tool-call' }]
})

const userText = (text: string): ChatMessage => ({
  id: 'u',
  role: 'user',
  parts: [{ text, type: 'text' }]
})

describe('skillTouchedInMessages', () => {
  it('detects a skill_view load of the skill', () => {
    expect(skillTouchedInMessages('pr-ready', [toolCall('skill_view', { name: 'pr-ready' })])).toBe(true)
  })

  it('detects a skill_manage touch of the skill', () => {
    expect(skillTouchedInMessages('pr-ready', [toolCall('skill_manage', { action: 'patch', name: 'pr-ready' })])).toBe(
      true
    )
  })

  it('matches qualified skill names (category/name, plugin:name)', () => {
    expect(skillTouchedInMessages('hermes-agent-dev', [toolCall('skill_view', { name: 'github/hermes-agent-dev' })])).toBe(
      true
    )
    expect(skillTouchedInMessages('writing-plans', [toolCall('skill_view', { name: 'superpowers:writing-plans' })])).toBe(
      true
    )
  })

  it('falls back to argsText when args were not parsed', () => {
    expect(skillTouchedInMessages('pr-ready', [toolCall('skill_view', undefined, '{"name":"pr-ready"}')])).toBe(true)
  })

  it('detects the user loading the skill via its slash command', () => {
    expect(skillTouchedInMessages('pr-ready', [userText('/pr-ready check this branch')])).toBe(true)
    expect(skillTouchedInMessages('pr-ready', [userText('/pr-ready')])).toBe(true)
  })

  it('ignores touches of OTHER skills and non-skill tools', () => {
    expect(skillTouchedInMessages('pr-ready', [toolCall('skill_view', { name: 'clean' })])).toBe(false)
    expect(skillTouchedInMessages('pr-ready', [toolCall('read_file', { path: 'pr-ready' })])).toBe(false)
    // Slash prefix must be exact — /pr-ready-extra is a different command.
    expect(skillTouchedInMessages('pr-ready', [userText('/pr-ready-extra go')])).toBe(false)
    // Merely mentioning the name in prose is not a load.
    expect(skillTouchedInMessages('pr-ready', [userText('is pr-ready any good?')])).toBe(false)
  })

  it('is case-insensitive on the stored arg', () => {
    expect(skillTouchedInMessages('pr-ready', [toolCall('skill_view', { name: 'PR-Ready' })])).toBe(true)
  })

  it('empty transcript touches nothing', () => {
    expect(skillTouchedInMessages('pr-ready', [])).toBe(false)
  })
})
