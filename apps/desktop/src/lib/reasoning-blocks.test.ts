import { describe, expect, it } from 'vitest'

import { separateGluedReasoningBlocks } from '@/lib/reasoning-blocks'

describe('separateGluedReasoningBlocks', () => {
  it('splits heading-onto-heading parts (the `****` run)', () => {
    const glued =
      '**Investigating likely culprit PRs****Inspecting message schema****Analyzing interrupted tool call impact**'

    expect(separateGluedReasoningBlocks(glued)).toBe(
      [
        '**Investigating likely culprit PRs**',
        '',
        '**Inspecting message schema**',
        '',
        '**Analyzing interrupted tool call impact**'
      ].join('\n')
    )
  })

  it('splits prose-onto-heading parts (vercel/ai#6742 repro)', () => {
    const glued =
      '**Simulating a greeting stream**\n\nIt feels like a streaming interaction!**Simulating a greeting stream**\n\nI want to meet the request.'

    expect(separateGluedReasoningBlocks(glued)).toContain('interaction!\n\n**Simulating')
    expect(separateGluedReasoningBlocks(glued)).not.toContain('interaction!**')
  })

  it('is idempotent on already-separated text', () => {
    const separated = '**One**\n\n**Two**'

    expect(separateGluedReasoningBlocks(separated)).toBe(separated)
  })

  it('leaves emphasis inside prose alone', () => {
    const prose = 'Looking at the logs, the **signature** field is missing — so the replay 400s.'

    expect(separateGluedReasoningBlocks(prose)).toBe(prose)
  })

  it('leaves an unclosed emphasis run alone', () => {
    expect(separateGluedReasoningBlocks('weighing options **')).toBe('weighing options **')
  })

  it('does not split a heading that already opens the text', () => {
    expect(separateGluedReasoningBlocks('**Only one part**')).toBe('**Only one part**')
  })

  // CJK prose sets no inter-word spaces, so the old lazy body matched from the
  // close of one bold pair to the open of the next and injected a paragraph
  // break mid-sentence (#107813). A match must be one complete pair that
  // closes its own line — inline emphasis is left exactly as written.
  it('leaves CJK inline bold mid-sentence alone (#107813)', () => {
    const prose = '1. **日经原因**（共同社）：指数重挫——**半导体领跌**——与美股同步。'

    expect(separateGluedReasoningBlocks(prose)).toBe(prose)
  })

  it('leaves punctuation-glued CJK emphasis alone (#107813)', () => {
    const prose = '——**唯一解释**：只有一条路——**破案**！'

    expect(separateGluedReasoningBlocks(prose)).toBe(prose)
  })

  it('leaves English inline bold mid-sentence alone', () => {
    const prose = 'The **core issue** is that the regex was over-broad.'

    expect(separateGluedReasoningBlocks(prose)).toBe(prose)
  })

  it('still splits a heading that closes its own line after prose', () => {
    const glued = 'interaction!**Checking logs**'

    expect(separateGluedReasoningBlocks(glued)).toBe('interaction!\n\n**Checking logs**')
  })

  it('still repairs the **First****Second** heading run', () => {
    expect(separateGluedReasoningBlocks('**First****Second**')).toBe('**First**\n\n**Second**')
  })

  it('is idempotent on separated CJK headings', () => {
    const separated = '**第一个部分**\n\n**第二个部分**'

    expect(separateGluedReasoningBlocks(separated)).toBe(separated)
  })
})
