import { describe, expect, it } from 'vitest'

import { preprocessMarkdown } from './markdown-preprocess'

describe('preprocessMarkdown file links with spaces (#102782)', () => {
  it('rewrites angle-bracket destinations that contain spaces into #preview hrefs', () => {
    const out = preprocessMarkdown('See [notes](<~/My Notes/todo with spaces.md>)')

    expect(out).toContain('#preview/')
    expect(out).toContain(encodeURIComponent('~/My Notes/todo with spaces.md'))
    expect(out).not.toMatch(/\]\(<~\/My Notes/)
  })

  it('still rewrites spaceless absolute and home-relative file links', () => {
    const out = preprocessMarkdown('[a](/tmp/a.md) and [b](~/b.md)')

    expect(out).toContain('#preview/')
    expect(out).toContain(encodeURIComponent('/tmp/a.md'))
    expect(out).toContain(encodeURIComponent('~/b.md'))
  })
})
