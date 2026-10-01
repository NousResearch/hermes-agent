import { describe, expect, it } from 'vitest'

import { preprocessMarkdown } from '@/lib/markdown-preprocess'

// Display math in the compact single-line `$$…$$` form: when the span owns its
// whole line it is promoted to the flow form (delimiters on their own lines)
// so remark-math parses it as display math and KaTeX centers it. Left alone,
// remark-math routes the single-line form through its inline mathText
// construct and the equation hugs the left edge of the bubble.
describe('single-line display math promotion', () => {
  it('promotes a lone $$…$$ line to delimiter-on-own-lines flow math', () => {
    expect(preprocessMarkdown('$$x^2$$')).toBe('$$\nx^2\n$$')
  })

  it('promotes when other prose blocks surround the math line', () => {
    const input = 'Intro paragraph.\n\n$$E = mc^2$$\n\nAfter the equation.'

    expect(preprocessMarkdown(input)).toBe('Intro paragraph.\n\n$$\nE = mc^2\n$$\n\nAfter the equation.')
  })

  it('keeps the container prefix (blockquote/list) on every emitted line', () => {
    expect(preprocessMarkdown('> $$x^2$$')).toBe('> $$\n> x^2\n> $$')
  })

  it('preserves CRLF line endings in the promoted form', () => {
    expect(preprocessMarkdown('$$x^2$$\r\n')).toBe('$$\r\nx^2\r\n$$\r\n')
  })

  it('leaves a mid-sentence $$…$$ inline', () => {
    const input = 'The value $$x^2$$ is positive.'

    expect(preprocessMarkdown(input)).toBe(input)
  })

  it('leaves a bare $$$$ alone', () => {
    expect(preprocessMarkdown('$$$$')).toBe('$$$$')
  })

  it('leaves a compact span with an embedded $$ alone', () => {
    expect(preprocessMarkdown('$$a $$ b$$')).toBe('$$a $$ b$$')
  })

  it('still splits the multi-line hugging form onto its own lines', () => {
    const input = '$$\\begin{aligned}\na &= b \\\\\nc &= d\n\\end{aligned}$$'

    expect(preprocessMarkdown(input)).toBe('$$\n\\begin{aligned}\na &= b \\\\\nc &= d\n\\end{aligned}\n$$')
  })
})
