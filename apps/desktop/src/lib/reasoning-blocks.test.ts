import { describe, expect, it } from 'vitest'

import { separateGluedReasoningBlocks } from './reasoning-blocks'

describe('separateGluedReasoningBlocks', () => {
  it('splits bare **** runs into separate paragraphs (heading-onto-heading)', () => {
    const input = '**First****Second**'
    const output = separateGluedReasoningBlocks(input)
    expect(output).toBe('**First**\n\n**Second**')
  })

  it('splits prose-glued heading onto its own line (prose-onto-heading)', () => {
    const input = 'interaction!**Checking logs**'
    const output = separateGluedReasoningBlocks(input)
    expect(output).toBe('interaction!\n\n**Checking logs**')
  })

  it('preserves inline bold when NOT at line-end (CJK case)', () => {
    const input = '1. **日経原因**（共同社）：指数重挫——**半導体領跌**——与美股同步。'
    const output = separateGluedReasoningBlocks(input)
    expect(output).toBe(input)
  })

  it('preserves inline bold mid-sentence (English)', () => {
    const input = 'The **core issue** is that the regex was over-broad.'
    const output = separateGluedReasoningBlocks(input)
    expect(output).toBe(input)
  })

  it('splits a line-final heading after prose', () => {
    const input = 'interaction!**Heading at end**'
    const output = separateGluedReasoningBlocks(input)
    expect(output).toBe('interaction!\n\n**Heading at end**')
  })

  it('leaves pre-separated blocks unchanged (idempotent)', () => {
    const input = '**First**\n\n**Second**'
    const output = separateGluedReasoningBlocks(input)
    expect(output).toBe(input)
  })
})
