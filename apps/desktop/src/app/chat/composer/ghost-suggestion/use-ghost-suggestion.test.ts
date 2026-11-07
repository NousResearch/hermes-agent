import { describe, expect, it } from 'vitest'

import { __testing, digestCatalog, parseCandidates } from './use-ghost-suggestion'

const { keywordFallback } = __testing

const SAMPLE_CATALOG = {
  pairs: [
    ['/commit', 'Create a git commit from staged changes'],
    ['/commit-push', 'Commit and push the current branch to remote'],
    ['/learn', 'Create a new skill from notes and examples'],
    ['/voice', 'Switch to voice input mode'],
    ['/gif-search', 'Search and send a reaction GIF']
  ],
  skills: {}
} as never

describe('parseCandidates', () => {
  it('returns nothing when the reply is NONE', () => {
    const valid = new Set(['/commit', '/learn'])
    expect(parseCandidates('NONE', valid)).toEqual([])
  })

  it('parses a single command per line', () => {
    const valid = new Set(['/commit', '/learn'])
    expect(parseCandidates('/learn\n/commit', valid)).toEqual([
      { command: '/learn', reason: '' },
      { command: '/commit', reason: '' }
    ])
  })

  it('parses the `command — reason` shorthand', () => {
    const valid = new Set(['/commit', '/learn'])
    const parsed = parseCandidates('/learn — matches the user asking to study', valid)
    expect(parsed).toHaveLength(1)
    expect(parsed[0]?.command).toBe('/learn')
    expect(parsed[0]?.reason).toBe('matches the user asking to study')
  })

  it('drops commands that are not in the catalog', () => {
    const valid = new Set(['/commit'])
    const parsed = parseCandidates('/learn\n/commit', valid)
    expect(parsed).toEqual([{ command: '/commit', reason: '' }])
  })

  it('caps the list at MAX_CANDIDATES (4)', () => {
    const valid = new Set(['/a', '/b', '/c', '/d', '/e'])
    const parsed = parseCandidates('/a\n/b\n/c\n/d\n/e', valid)
    expect(parsed.map(c => c.command)).toEqual(['/a', '/b', '/c', '/d'])
  })

  it('skips blank lines and ignores commands without a leading slash', () => {
    const valid = new Set(['/commit'])
    const parsed = parseCandidates('\n  \ncommit\n/commit\n', valid)
    expect(parsed).toEqual([{ command: '/commit', reason: '' }])
  })

  it('deduplicates repeated commands', () => {
    const valid = new Set(['/commit'])
    const parsed = parseCandidates('/commit\n/commit\n/commit', valid)
    expect(parsed).toEqual([{ command: '/commit', reason: '' }])
  })
})

describe('keywordFallback', () => {
  const valid = new Set(['/learn', '/commit', '/commit-push', '/voice', '/gif-search'])

  it('matches a Chinese "学习" draft to /learn', () => {
    const result = keywordFallback('想要学习无人机CAAC证书', valid)

    expect(result.length).toBeGreaterThan(0)
    expect(result[0]?.command).toBe('/learn')
  })

  it('matches a Chinese "提交" draft to /commit', () => {
    const result = keywordFallback('帮我提交代码', valid)

    expect(result[0]?.command).toBe('/commit')
  })

  it('matches "push" before "commit" when the draft mentions pushing', () => {
    const result = keywordFallback('push my changes', valid)

    expect(result.map(r => r.command)).toContain('/commit-push')
  })

  it('returns an empty list when no keyword matches', () => {
    expect(keywordFallback('random unrelated text', valid)).toEqual([])
  })

  it('drops candidates that are not in the live catalog', () => {
    const result = keywordFallback('学习 caac', new Set(['/commit'])) // /learn missing

    expect(result).toEqual([])
  })

  it('returns the candidates in score-descending order', () => {
    // "学习 caac 无人机" should produce /learn with the strongest score
    const result = keywordFallback('学习 caac 无人机', valid)
    expect(result[0]?.command).toBe('/learn')
  })
})

describe('digestCatalog', () => {
  it('returns an empty string for a null catalog', () => {
    expect(digestCatalog(null)).toBe('')
    expect(digestCatalog(undefined)).toBe('')
  })

  it('serializes every pair into a `command: description` line', () => {
    const digest = digestCatalog(SAMPLE_CATALOG)
    expect(digest).toContain('/learn: Create a new skill from notes and examples')
    expect(digest).toContain('/voice: Switch to voice input mode')
    expect(digest).toContain('/gif-search: Search and send a reaction GIF')
  })

  it('normalizes commands without a leading slash', () => {
    const digest = digestCatalog({ pairs: [['learn', 'study a topic']], skills: {} } as never)
    expect(digest).toBe('/learn: study a topic')
  })

  it('falls back to a "(no description)" marker for empty descriptions', () => {
    const digest = digestCatalog({ pairs: [['/stub', '']], skills: {} } as never)
    expect(digest).toBe('/stub: (no description)')
  })
})
