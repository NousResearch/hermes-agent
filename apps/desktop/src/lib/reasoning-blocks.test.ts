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

  it('leaves CJK mid-sentence inline bold unchanged (issue #107813)', () => {
    const cjk = '1. **日经原因**（共同社）：指数重挫——**半导体领跌**——与美股同步。'

    expect(separateGluedReasoningBlocks(cjk)).toBe(cjk)
  })

  it('leaves CJK punctuation-glued inline bold that does not close the line unchanged', () => {
    const glued = '——**唯一解释**：后续——**破案**！'

    expect(separateGluedReasoningBlocks(glued)).toBe(glued)
  })

  it('splits a line-final heading glued onto prose', () => {
    expect(separateGluedReasoningBlocks('interaction!**Checking logs**')).toBe(
      'interaction!\n\n**Checking logs**'
    )
  })

  it('splits heading-onto-heading First/Second control', () => {
    expect(separateGluedReasoningBlocks('**First****Second**')).toBe('**First**\n\n**Second**')
  })

  it('does not treat a bold pair containing inner * as a glued heading', () => {
    const nested = 'prose!**foo *bar* baz**'

    expect(separateGluedReasoningBlocks(nested)).toBe(nested)
  })
})
