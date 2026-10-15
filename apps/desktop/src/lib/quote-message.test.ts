import { describe, expect, it } from 'vitest'

import { formatQuoteBlock } from './quote-message'

describe('formatQuoteBlock', () => {
  it('wraps a single-line body in a markdown blockquote with role + timestamp header', () => {
    const result = formatQuoteBlock('user', 'hello world', '10:32')

    expect(result).toBe('> [user 10:32]\n> hello world\n\n')
  })

  it('preserves multi-line bodies, prefixing every line with the quote marker', () => {
    const result = formatQuoteBlock('assistant', 'first\nsecond\nthird', '14:05')

    expect(result).toBe('> [assistant 14:05]\n> first\n> second\n> third\n\n')
  })

  it('escapes bodies that already start with > so the quote block stays open', () => {
    const result = formatQuoteBlock('user', '> already a quote\nnext', '09:00')

    // The leading > on the first body line is intentional markdown; we keep
    // it readable by indenting one space, which concatenates inside the
    // outer blockquote rather than closing it.
    expect(result).toBe('> [user 09:00]\n>  > already a quote\n> next\n\n')
  })

  it('falls back to [empty message] when the body is whitespace-only', () => {
    const result = formatQuoteBlock('user', '   \n\t  ', '00:00')

    expect(result).toBe('> [user 00:00]\n> [empty message]\n\n')
  })

  it('trims trailing whitespace so the trailing paragraph break stays clean', () => {
    const result = formatQuoteBlock('assistant', 'real text   \n   \n', '23:59')

    expect(result.endsWith('\n\n')).toBe(true)
    expect(result).not.toContain('real text   \n')
  })
})
