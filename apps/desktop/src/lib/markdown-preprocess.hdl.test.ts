import { describe, expect, it } from 'vitest'

import { preprocessMarkdown } from './markdown-preprocess'

// Verilog/SystemVerilog system functions are bare `$` + identifier. With
// `singleDollarTextMath: true`, two of them on one prose line open an inline
// math span and the intervening HDL is typeset through KaTeX — `<=`, the
// non-blocking assignment operator, renders as `≤` and identifiers go
// math-italic, so the reader is shown wrong HDL (#129491).
describe('HDL system-function dollars', () => {
  it('escapes both dollars of an HDL pair on one prose line', () => {
    const out = preprocessMarkdown('if ($bits(a) <= $bits(b)) ok = 1;')

    expect(out).toContain('\\$bits(a) <= \\$bits(b)')
  })

  it('escapes every system function in a longer one-liner', () => {
    const out = preprocessMarkdown('assign y = $random; and $clog2 for width.')

    expect(out).toContain('\\$random; and \\$clog2')
  })

  it('leaves real inline math untouched', () => {
    expect(preprocessMarkdown('$x + y$')).toBe('$x + y$')
    expect(preprocessMarkdown('$f(x) <= g(x)$')).toBe('$f(x) <= g(x)$')
  })

  it('escapes a lone system function that has no same-line partner', () => {
    expect(preprocessMarkdown('Call $display to trace the value.')).toContain('\\$display')
  })

  it('does not double-escape an already escaped dollar', () => {
    expect(preprocessMarkdown('already \\$bits here')).toBe('already \\$bits here')
  })
})