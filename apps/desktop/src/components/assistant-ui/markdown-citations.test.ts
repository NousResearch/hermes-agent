import { describe, expect, it } from 'vitest'

import { preprocessMarkdown } from '@/lib/markdown-preprocess'

describe('grounded citation preprocessing', () => {
  it('preserves inline markers that map to the Sources section', () => {
    const input = ['Supported claim.[1]', '', '## Sources', '', '[1] https://example.com/source — Source'].join('\n')

    const output = preprocessMarkdown(input)

    expect(output).toContain('Supported claim.[1]')
    expect(output).toContain('[1] <https://example.com/source>')
  })

  it('preserves adjacent mapped markers from the bundled skill format', () => {
    const input = [
      'Supported by two sources.[1][2]',
      '',
      'Sources:',
      '[1] https://example.com/one',
      '[2] https://example.com/two'
    ].join('\n')

    const output = preprocessMarkdown(input)

    expect(output).toContain('Supported by two sources.[1][2]')
  })

  it('removes a marker whose id is missing from a real Sources section', () => {
    const input = ['Claim with the wrong source.[2]', '', '## Sources', '', '[1] https://example.com/one'].join('\n')

    const output = preprocessMarkdown(input)

    expect(output).toContain('Claim with the wrong source.')
    expect(output).not.toContain('Claim with the wrong source.[2]')
  })

  it('still removes orphan numeric markers', () => {
    const output = preprocessMarkdown('This is the source[0], but keep `items[0]` untouched.')

    expect(output).toContain('source,')
    expect(output).not.toContain('source[0]')
    expect(output).toContain('`items[0]`')
  })

  it('does not trust source entries that only appear inside a code fence', () => {
    const input = [
      'Unsupported claim.[1]',
      '',
      '```text',
      '## Sources',
      '[1] https://example.com/not-a-real-source-block',
      '```'
    ].join('\n')

    const output = preprocessMarkdown(input)

    expect(output).toContain('Unsupported claim.')
    expect(output).not.toContain('Unsupported claim.[1]')
  })
})
