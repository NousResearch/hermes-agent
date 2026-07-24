import { describe, expect, it } from 'vitest'

import { preprocessMarkdown } from './markdown-preprocess'

describe('preprocessMarkdown text fences', () => {
  it('preserves explicit text fences instead of leaking the language tag into prose', () => {
    const input = [
      '```text',
      'apps/desktop/src/components/assistant-ui/thread.tsx',
      'apps/desktop/src/lib/markdown-code.ts',
      'apps/desktop/src/styles.css',
      '```'
    ].join('\n')

    expect(preprocessMarkdown(input)).toBe(input)
  })

  it('preserves explicit plain text fences', () => {
    const input = ['```plaintext', 'hello world', '```'].join('\n')

    expect(preprocessMarkdown(input)).toBe(input)
  })
})
