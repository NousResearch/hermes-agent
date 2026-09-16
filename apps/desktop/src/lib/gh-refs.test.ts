import { describe, expect, it } from 'vitest'

import { ghIssueUrl, ghRefFromMarkdownHref, linkifyGhIssueRefs } from './gh-refs'

describe('linkifyGhIssueRefs', () => {
  it('links a bare ref in prose', () => {
    expect(linkifyGhIssueRefs('merga PR #46')).toBe('merga PR [#46](#gh-ref/46)')
  })

  it('links multiple refs in one line', () => {
    expect(linkifyGhIssueRefs('fixes #12 and #345 today')).toBe(
      'fixes [#12](#gh-ref/12) and [#345](#gh-ref/345) today'
    )
  })

  it('keeps trailing prose punctuation outside the link', () => {
    expect(linkifyGhIssueRefs('landed in #46.')).toBe('landed in [#46](#gh-ref/46).')
    expect(linkifyGhIssueRefs('#12, and #13')).toBe('[#12](#gh-ref/12), and [#13](#gh-ref/13)')
  })

  it('never touches an ATX heading', () => {
    expect(linkifyGhIssueRefs('# Title')).toBe('# Title')
    expect(linkifyGhIssueRefs('## Section')).toBe('## Section')
    expect(linkifyGhIssueRefs('#46')).toBe('[#46](#gh-ref/46)')
  })

  it('does not swallow an emphasis run abutting the ref', () => {
    // **#46** — a heading-shaped token glued to emphasis must still link.
    expect(linkifyGhIssueRefs('**#46** today')).toBe('**[#46](#gh-ref/46)** today')
  })

  it('skips refs already wrapped in a markdown link', () => {
    expect(linkifyGhIssueRefs('[see #46](https://x) ')).toBe('[see #46](https://x) ')
    expect(linkifyGhIssueRefs('[#46](#gh-ref/46)')).toBe('[#46](#gh-ref/46)')
  })

  it('does not nest a ref inside a labeled link (the agent’s own PR-link format)', () => {
    expect(linkifyGhIssueRefs('[PR #46](https://github.com/o/r/pull/46)')).toBe(
      '[PR #46](https://github.com/o/r/pull/46)'
    )
  })

  it('skips refs inside url fragments already autolinked upstream', () => {
    expect(linkifyGhIssueRefs('<https://x.dev/a#46> next')).toBe('<https://x.dev/a#46> next')
  })

  it('skips refs glued to words or other hashes', () => {
    expect(linkifyGhIssueRefs('abc#12')).toBe('abc#12')
    expect(linkifyGhIssueRefs('##46')).toBe('##46')
    expect(linkifyGhIssueRefs('a#12')).toBe('a#12')
  })

  it('skips footnote-style tokens', () => {
    expect(linkifyGhIssueRefs('text[^1] more')).toBe('text[^1] more')
  })

  it('rejects more than 7 digits', () => {
    expect(linkifyGhIssueRefs('#12345678')).toBe('#12345678')
  })

  it('accepts 7-digit refs', () => {
    expect(linkifyGhIssueRefs('#1234567 ok')).toBe('[#1234567](#gh-ref/1234567) ok')
  })

  it('is a no-op without a hash', () => {
    expect(linkifyGhIssueRefs('no refs here')).toBe('no refs here')
  })
})

describe('ghRefFromMarkdownHref', () => {
  it('parses valid refs', () => {
    expect(ghRefFromMarkdownHref('#gh-ref/46')).toBe(46)
    expect(ghRefFromMarkdownHref('#gh-ref/1234567')).toBe(1234567)
  })

  it('rejects everything else', () => {
    expect(ghRefFromMarkdownHref('#gh-ref/12345678')).toBeNull()
    expect(ghRefFromMarkdownHref('#gh-ref/abc')).toBeNull()
    expect(ghRefFromMarkdownHref('#gh-ref/')).toBeNull()
    expect(ghRefFromMarkdownHref('#session/abc')).toBeNull()
    expect(ghRefFromMarkdownHref(undefined)).toBeNull()
  })
})

describe('ghIssueUrl', () => {
  it('builds the github.com URL', () => {
    expect(ghIssueUrl('KaptenKatthatt', 'newsAgg', 46)).toBe('https://github.com/KaptenKatthatt/newsAgg/issues/46')
  })
})