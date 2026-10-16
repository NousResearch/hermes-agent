import { describe, expect, it } from 'vitest'

import { clampHtmlNestingDepth } from './markdown-html-depth'

// The clamp exists to keep parse5 / hast-util-from-parse5 recursion off the
// call stack. Its contract is about DEPTH of unclosed elements, so the tests
// assert that relationship rather than any particular output string.
describe('clampHtmlNestingDepth', () => {
  it('returns text without tags unchanged, by identity', () => {
    const text = 'C:\\Windows\\System32 is 12 GB, and 5 < 7 is true'

    expect(clampHtmlNestingDepth(text)).toBe(text)
  })

  it('leaves ordinary markup untouched, by identity', () => {
    const text = '<div class="a"><p>A <b>bold</b> claim</p></div>'

    expect(clampHtmlNestingDepth(text)).toBe(text)
  })

  it('leaves a wide but shallow tag run untouched — count is not the risk', () => {
    const text = '<b>x</b>'.repeat(20_000)

    expect(clampHtmlNestingDepth(text)).toBe(text)
  })

  it('leaves void elements untouched — they never add depth', () => {
    const text = '<br>'.repeat(20_000)

    expect(clampHtmlNestingDepth(text)).toBe(text)
  })

  it('leaves self-closing tags untouched — they never add depth', () => {
    const text = '<custom-el />'.repeat(20_000)

    expect(clampHtmlNestingDepth(text)).toBe(text)
  })

  it('escapes an unbounded run of unclosed unknown tags', () => {
    const clamped = clampHtmlNestingDepth('Let' + '<unk>'.repeat(16_383))

    // Nothing is dropped: the text is still all there, just literal. Only the
    // opening `<` needs escaping — that alone stops parse5 opening an element.
    expect(clamped).toContain('&lt;unk>')
    expect(clamped.startsWith('Let<unk>')).toBe(true)
  })

  it('escapes an unbounded run of unclosed known tags', () => {
    const clamped = clampHtmlNestingDepth('<div>'.repeat(5_000))

    expect(clamped).toContain('&lt;div>')
  })

  it('keeps the prefix below the cap parseable and only escapes past it', () => {
    const clamped = clampHtmlNestingDepth('<unk>'.repeat(5_000))
    const parseable = clamped.slice(0, clamped.indexOf('&lt;'))

    // Everything before the overflow point is untouched real markup...
    expect(parseable).toBe('<unk>'.repeat(parseable.length / '<unk>'.length))
    // ...and it is bounded, which is the entire point.
    expect(parseable.length).toBeLessThan('<unk>'.repeat(5_000).length)
  })

  it('does not accumulate depth when tags close, however many there are', () => {
    const text = '<div>x</div>'.repeat(10_000)

    expect(clampHtmlNestingDepth(text)).toBe(text)
  })
})
