import { describe, expect, it } from 'vitest'

import { preprocessMarkdown } from './markdown-preprocess'
import {
  linkifySessionRefs,
  parseSessionRefValue,
  sessionMarkdownHref,
  sessionRefCacheKey,
  sessionRefFallbackLabel,
  sessionRefFromMarkdownHref,
  splitSessionRefValue
} from './session-refs'

describe('parseSessionRefValue', () => {
  it('splits profile and session id', () => {
    expect(parseSessionRefValue('work/20260101_abc123')).toEqual({
      profile: 'work',
      sessionId: '20260101_abc123'
    })
  })

  it('treats bare values as session ids', () => {
    expect(parseSessionRefValue('20260101_abc123')).toEqual({ sessionId: '20260101_abc123' })
  })
})

describe('sessionRefFallbackLabel', () => {
  it('truncates long ids', () => {
    expect(sessionRefFallbackLabel('default/20260610_120000_abcdef')).toBe('20260610…')
  })

  it('leaves short ids alone', () => {
    expect(sessionRefFallbackLabel('work/abc123')).toBe('abc123')
  })
})

describe('sessionRefCacheKey', () => {
  it('separates the same id across profiles', () => {
    expect(sessionRefCacheKey('work/abc')).not.toBe(sessionRefCacheKey('home/abc'))
  })

  it('is empty for a valueless ref', () => {
    expect(sessionRefCacheKey('  ')).toBe('')
  })
})

describe('splitSessionRefValue', () => {
  it('peels prose punctuation off a bare value', () => {
    expect(splitSessionRefValue('default/abc123.')).toEqual({ trailing: '.', value: 'default/abc123' })
  })

  it('leaves a quoted value fenced', () => {
    expect(splitSessionRefValue('`my session`')).toEqual({ trailing: '', value: 'my session' })
  })
})

describe('session markdown hrefs', () => {
  it('round-trips a value through the fragment href', () => {
    const href = sessionMarkdownHref('work/20260101_abc123')

    expect(href).toBe('#session/work%2F20260101_abc123')
    expect(sessionRefFromMarkdownHref(href)).toBe('work/20260101_abc123')
  })

  it('ignores hrefs that are not session fragments', () => {
    expect(sessionRefFromMarkdownHref('#preview/foo')).toBeNull()
    expect(sessionRefFromMarkdownHref('https://example.com')).toBeNull()
    expect(sessionRefFromMarkdownHref(undefined)).toBeNull()
  })
})

describe('linkifySessionRefs', () => {
  it('rewrites a bare ref into a session link', () => {
    expect(linkifySessionRefs('see @session:work/20260101_abc123 next')).toBe(
      'see [20260101…](#session/work%2F20260101_abc123) next'
    )
  })

  it('keeps trailing prose punctuation outside the link', () => {
    expect(linkifySessionRefs('see @session:default/abc123.')).toBe('see [abc123](#session/default%2Fabc123).')
  })

  it('leaves text without a ref untouched', () => {
    const text = 'no references here'

    expect(linkifySessionRefs(text)).toBe(text)
  })

  it('does not match an email-like or path-embedded @session', () => {
    expect(linkifySessionRefs('me@session:abc')).toBe('me@session:abc')
    expect(linkifySessionRefs('a/@session:abc')).toBe('a/@session:abc')
  })

  it('leaves a ref a model already wrapped in a markdown link alone', () => {
    const text = '[that chat](@session:work/abc123)'

    expect(linkifySessionRefs(text)).toBe(text)
  })
})

// The agent-authored path: assistant markdown runs through preprocessMarkdown
// before Streamdown, so a bare ref must survive as a link — and must NOT be
// rewritten inside code, where it is being discussed rather than referenced.
describe('preprocessMarkdown session refs', () => {
  it('linkifies a ref in prose', () => {
    expect(preprocessMarkdown('Context is in @session:work/20260101_abc123 there.')).toContain(
      '[20260101…](#session/work%2F20260101_abc123)'
    )
  })

  it('leaves refs inside inline code alone', () => {
    const out = preprocessMarkdown('Type `@session:work/20260101_abc123` to link a chat.')

    expect(out).toContain('`@session:work/20260101_abc123`')
    expect(out).not.toContain('#session/')
  })

  it('leaves refs inside a fenced block alone', () => {
    const out = preprocessMarkdown(['```text', '@session:work/20260101_abc123', '```'].join('\n'))

    expect(out).toContain('@session:work/20260101_abc123')
    expect(out).not.toContain('#session/')
  })
})
