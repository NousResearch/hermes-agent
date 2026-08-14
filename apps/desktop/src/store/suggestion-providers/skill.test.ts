import { describe, expect, it } from 'vitest'

import { collidesWithWorkspace, skillHit, skillPattern } from './skill'

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
