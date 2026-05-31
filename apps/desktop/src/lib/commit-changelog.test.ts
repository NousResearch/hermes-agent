import { describe, expect, it } from 'vitest'

import { buildCommitChangelog, parseCommitHeader } from './commit-changelog'

describe('parseCommitHeader', () => {
  it('extracts type, scope, and subject from a conventional header', () => {
    expect(parseCommitHeader('feat(desktop): NSIS prereq detection page')).toEqual({
      breaking: false,
      scope: 'desktop',
      subject: 'NSIS prereq detection page',
      type: 'feat'
    })
  })

  it('flags breaking changes via the `!` marker', () => {
    expect(parseCommitHeader('feat(api)!: change endpoint shape')).toMatchObject({
      breaking: true,
      type: 'feat'
    })
  })

  it('treats non-conventional commits as untyped with the full header as subject', () => {
    expect(parseCommitHeader('Update README')).toEqual({
      breaking: false,
      scope: null,
      subject: 'Update README',
      type: null
    })
  })

  it('ignores body lines and trims whitespace', () => {
    expect(parseCommitHeader('  fix: handle null input  \n\nMore detail')).toMatchObject({
      subject: 'handle null input',
      type: 'fix'
    })
  })

  it('returns empty subject for blank input', () => {
    expect(parseCommitHeader('')).toEqual({ breaking: false, scope: null, subject: '', type: null })
  })
})

describe('buildCommitChangelog', () => {
  it('groups commits into user-friendly buckets and capitalizes subjects', () => {
    const groups = buildCommitChangelog([
      { summary: 'feat(desktop): add NSIS prereq detection page' },
      { summary: 'fix(sidebar): jitter when dragging' },
      { summary: 'perf: shave 200ms off cold start' },
      { summary: 'refactor: extract sidebar row component' }
    ])

    expect(groups.map(g => g.id)).toEqual(['new', 'fixed', 'faster'])
    expect(groups[0]).toMatchObject({ label: "What's new" })
    expect(groups[0].items[0]).toBe('Add NSIS prereq detection page')
    expect(groups[1].items[0]).toBe('Jitter when dragging')
  })

  it('hides chore/ci/docs/test commits', () => {
    const groups = buildCommitChangelog([
      { summary: 'chore: bump deps' },
      { summary: 'ci: tweak workflow' },
      { summary: 'docs: spelling fix' },
      { summary: 'feat: real new feature' }
    ])

    expect(groups).toHaveLength(1)
    expect(groups[0].items).toEqual(['Real new feature'])
  })

  it('routes unparseable commits to the "Other improvements" bucket', () => {
    const groups = buildCommitChangelog([{ summary: 'Update sidebar styling' }])

    expect(groups[0].id).toBe('other')
    expect(groups[0].items).toEqual(['Update sidebar styling'])
  })

  it('falls back to a neutral placeholder when every commit is filtered or empty', () => {
    const groups = buildCommitChangelog([{ summary: 'chore: bump' }, { summary: 'ci: stuff' }])

    expect(groups).toEqual([{ id: 'other', items: ['Improvements and fixes'], label: 'In this update' }])
  })

  it('dedupes identical subjects and caps the items per group', () => {
    const groups = buildCommitChangelog(
      [
        { summary: 'fix: thing A' },
        { summary: 'fix: thing A' },
        { summary: 'fix: thing B' },
        { summary: 'fix: thing C' },
        { summary: 'fix: thing D' },
        { summary: 'fix: thing E' }
      ],
      { maxPerGroup: 3, maxTotal: 10 }
    )

    expect(groups[0].items).toEqual(['Thing A', 'Thing B', 'Thing C'])
  })

  it('caps total entries across buckets', () => {
    const groups = buildCommitChangelog(
      [
        { summary: 'feat: a' },
        { summary: 'feat: b' },
        { summary: 'fix: c' },
        { summary: 'fix: d' },
        { summary: 'perf: e' }
      ],
      { maxTotal: 3 }
    )

    const totalItems = groups.reduce((sum, g) => sum + g.items.length, 0)
    expect(totalItems).toBe(3)
  })
})
