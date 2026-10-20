import { describe, expect, it } from 'vitest'

import { formatRefValue, hermesDirectiveFormatter } from './directive-text'

describe('formatRefValue', () => {
  it('leaves simple paths untouched', () => {
    expect(formatRefValue('src/index.ts')).toBe('src/index.ts')
    expect(formatRefValue('https://example.com/post')).toBe('https://example.com/post')
  })

  it('wraps paths with whitespace in backticks', () => {
    expect(formatRefValue('apple-touch-icon (1).png')).toBe('`apple-touch-icon (1).png`')
  })

  it('falls back to double quotes when value contains backticks', () => {
    expect(formatRefValue('weird `name` (1).md')).toBe('"weird `name` (1).md"')
  })
})

describe('hermesDirectiveFormatter.parse', () => {
  it('keeps quoted file paths whole when parsing', () => {
    const segments = hermesDirectiveFormatter.parse('see @image:`apple-touch-icon (1).png` for the icon')

    expect(segments).toEqual([
      { kind: 'text', text: 'see ' },
      { kind: 'mention', type: 'image', label: 'apple-touch-icon (1).png', id: 'apple-touch-icon (1).png' },
      { kind: 'text', text: ' for the icon' }
    ])
  })

  it('still parses unquoted paths', () => {
    const segments = hermesDirectiveFormatter.parse('@file:src/main.tsx the entry point')

    // The label keeps its directory: it's the same string the `@` popover row
    // showed, and a bare `main.tsx` can't tell two files apart.
    expect(segments).toEqual([
      { kind: 'mention', type: 'file', label: 'src/main.tsx', id: 'src/main.tsx' },
      { kind: 'text', text: ' the entry point' }
    ])
  })

  it('preserves punctuation inside quoted values but trims bare sentence punctuation', () => {
    const quoted = hermesDirectiveFormatter.parse('see @file:`report!` now')
    const bare = hermesDirectiveFormatter.parse('see @file:report! now')

    expect(quoted).toEqual([
      { kind: 'text', text: 'see ' },
      { kind: 'mention', type: 'file', label: 'report!', id: 'report!' },
      { kind: 'text', text: ' now' }
    ])
    expect(bare).toEqual([
      { kind: 'text', text: 'see ' },
      { kind: 'mention', type: 'file', label: 'report', id: 'report' },
      { kind: 'text', text: ' now' }
    ])
  })

  it('parses session links with profile/id values', () => {
    const segments = hermesDirectiveFormatter.parse('see @session:work/20260101_abc123 next')

    expect(segments).toEqual([
      { kind: 'text', text: 'see ' },
      { kind: 'mention', type: 'session', label: '20260101…', id: 'work/20260101_abc123' },
      { kind: 'text', text: ' next' }
    ])
  })
})

describe('inline skill references', () => {
  const skills = (text: string) =>
    [...hermesDirectiveFormatter.parse(text)]
      .filter(segment => segment.kind === 'mention' && segment.type === 'skill')
      .map(segment => (segment.kind === 'mention' ? segment.id : ''))

  it('keeps a picked skill a chip in the sent message instead of flattening it', () => {
    expect(skills('please run /clean on this')).toEqual(['/clean'])
  })

  it('keeps the surrounding prose as text around the chip', () => {
    const segments = hermesDirectiveFormatter.parse('tidy this with /clean thanks')

    expect(segments).toEqual([
      { kind: 'text', text: 'tidy this with ' },
      { kind: 'mention', type: 'skill', label: 'clean', id: '/clean' },
      { kind: 'text', text: ' thanks' }
    ])
  })

  it('leaves file paths and fractions alone', () => {
    expect(skills('check src/foo/bar')).toEqual([])
    expect(skills('look at /usr/local/bin')).toEqual([])
    expect(skills('roughly 3 /4 of it')).toEqual([])
  })

  it('chips a leading slash, which now reaches the transcript as a skill invocation', () => {
    // #71664 asserted the opposite, and was right at the time: a leading slash
    // only ever EXECUTED, so it never reached a rendered message as text —
    // the turn that reached the bubble was the expanded skill body. Projecting
    // a skill turn back onto `/work fix it` changes that precondition, so the
    // invocation now has to chip like any other skill reference.
    expect(skills('/clean')).toEqual(['/clean'])
    expect(skills('/work fix the leak')).toEqual(['/work'])
  })

  it('parses a skill chip alongside an @ reference', () => {
    const mentions = [...hermesDirectiveFormatter.parse('run /clean on @file:`src/a.ts`')].filter(
      segment => segment.kind === 'mention'
    )

    expect(mentions.map(segment => (segment.kind === 'mention' ? segment.type : ''))).toEqual(['skill', 'file'])
  })
})
