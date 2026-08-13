import { describe, expect, it } from 'vitest'

import { skillPattern } from './skill'

describe('skillPattern', () => {
  it('matches the exact name as a whole word', () => {
    expect(skillPattern('perf').test('run the perf loop')).toBe(true)
    expect(skillPattern('perf').test('performance is bad')).toBe(false)
  })

  it('hyphenated names also match spaced phrasing', () => {
    expect(skillPattern('pr-ready').test('make this pr ready')).toBe(true)
    expect(skillPattern('pr-ready').test('run pr-ready on it')).toBe(true)
    expect(skillPattern('pr_ready').test('pr ready please')).toBe(true)
  })

  it('never matches inside other words', () => {
    expect(skillPattern('read').test('i already did')).toBe(false)
    expect(skillPattern('work').test('reworked the layout')).toBe(false)
    // Suffix boundary includes hyphen: "clean" must not fire inside "clean-up"
    // (a different skill may own that name).
    expect(skillPattern('clean').test('do a clean-up pass')).toBe(false)
  })

  it('is case-insensitive via lowercased input', () => {
    expect(skillPattern('Clean').test('please clean this diff')).toBe(true)
  })
})
