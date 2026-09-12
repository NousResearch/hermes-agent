import { describe, expect, it } from 'vitest'

import type { ChatMessage } from '@/lib/chat-messages'

import { collidesWithWorkspace, skillHit, skillPattern, skillTouchedInMessages, stripSlashTokens } from './skill'

// skillHit is the provider's real predicate: a whole-word match that the user
// has finished typing (at least one character follows it).
const hits = (name: string, draft: string) => skillHit(skillPattern(name), draft.toLowerCase())

// Exercises the provider's real haystack sanitizer, not a copy of it.
const hitsAfterSanitize = (name: string, draft: string) =>
  skillHit(skillPattern(name), stripSlashTokens(draft).toLowerCase())

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
    expect(
      skillTouchedInMessages('hermes-agent-dev', [toolCall('skill_view', { name: 'github/hermes-agent-dev' })])
    ).toBe(true)
    expect(
      skillTouchedInMessages('writing-plans', [toolCall('skill_view', { name: 'superpowers:writing-plans' })])
    ).toBe(true)
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

describe('draft haystack sanitization (slash-prefixed skills)', () => {
  it('strips slash commands mid-message before scanning', () => {
    // The bug: "please run /github-auth on this" triggered "Use skill: github-auth"
    // because skillHit matched "github-auth" inside the slash command.
    expect(hitsAfterSanitize('github-auth', 'please run /github-auth on this')).toBe(false)
    expect(hitsAfterSanitize('vault', 'When I trigger /vault and /github-auth please')).toBe(false)
  })

  it('preserves URL fragments mid-prose (no leading whitespace)', () => {
    // https://example.com/api/v1 is not stripped because there's no \s before /api.
    // The regex only strips whitespace-bounded slash commands.
    expect(hitsAfterSanitize('api', 'visit https://example.com/api/v1 for api docs ')).toBe(true)
  })

  it('a real prose mention after a slash command still hits', () => {
    // Once a real prose mention of github-auth appears after the slash command,
    // that mention still hits (the slash token is stripped, the prose mention remains).
    expect(hitsAfterSanitize('github-auth', '/github-auth loaded. Is github-auth any good? ')).toBe(true)
  })
})

