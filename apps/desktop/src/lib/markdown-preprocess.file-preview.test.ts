import { describe, expect, it } from 'vitest'

import { normalizeFilePreviewMath } from '@/lib/markdown-preprocess'

describe('normalizeFilePreviewMath', () => {
  it('leaves single-dollar inline math untouched', () => {
    expect(normalizeFilePreviewMath('Price is $x^2 + y^2$ today')).toBe('Price is $x^2 + y^2$ today')
  })

  it('escapes currency dollar amounts so they are not parsed as math', () => {
    const output = normalizeFilePreviewMath('Rent $500 and utilities $150.')

    expect(output).toContain('\\$500')
    expect(output).toContain('\\$150')
  })

  it('does not escape dollars inside code fences', () => {
    const input = ['Before', '```bash', 'echo $HOME', 'echo $PATH', '```', 'After'].join('\n')

    expect(normalizeFilePreviewMath(input)).toBe(input)
  })

  it('does not escape dollars inside inline code spans', () => {
    expect(normalizeFilePreviewMath('Run `echo $HOME` now')).toBe('Run `echo $HOME` now')
  })

  it('normalizes latex delimiters to dollar math', () => {
    // `\(…\)` and `\[…\]` are rewritten to `$…$` / `$$…$$` so remark-math parses them.
    expect(normalizeFilePreviewMath('A formula \\(x + 1\\) inline.')).toBe('A formula $x + 1$ inline.')
  })

  it('leaves ```math fences untouched (rehype renders language-math directly)', () => {
    const input = ['```math', 'E = mc^2', '```'].join('\n')

    expect(normalizeFilePreviewMath(input)).toBe(input)
  })

  it('does not strip citation markers or autolink urls (chat-only transforms omitted)', () => {
    const input = 'See [3] and https://example.com'

    const output = normalizeFilePreviewMath(input)

    // A chat message would autolink the URL and strip the [3] citation marker;
    // a file's prose must stay verbatim.
    expect(output).toBe(input)
  })

  it('does not strip reasoning blocks (author content, not model output)', () => {
    const input = '<reasoning>keep me</reasoning>'

    expect(normalizeFilePreviewMath(input)).toBe(input)
  })
})
