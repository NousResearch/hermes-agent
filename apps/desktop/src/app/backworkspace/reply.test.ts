import { describe, expect, it } from 'vitest'

import { replyBlock, replyInsertion } from './reply'

const AT = new Date('2026-09-20T14:05:00')

describe('replyBlock', () => {
  it('quotes every line under the name that answered', () => {
    const block = replyBlock('@hermes', 'first line\n\nthird line', AT)
    const [head, ...body] = block.split('\n')

    expect(head).toMatch(/^> hermes · /)
    expect(body).toEqual(['> first line', '>', '> third line'])
  })
})

describe('replyInsertion', () => {
  it('puts the reply right after the question it answers', () => {
    const doc = 'a note\n\n@hermes what is this?\n\nmore writing'
    const { from, insert } = replyInsertion(doc, '@hermes what is this?', '> hermes · 14:05\n> an answer')

    expect(doc.slice(0, from)).toBe('a note\n\n@hermes what is this?')
    // The line below is already a blank line away — nothing is added to it.
    expect(insert).toBe('\n\n> hermes · 14:05\n> an answer')
  })

  it('keeps writing that follows on the next line clear of the reply', () => {
    const doc = '@hermes what is this?\nthe next line'
    const { insert } = replyInsertion(doc, '@hermes what is this?', 'reply')

    expect(insert).toBe('\n\nreply\n\n')
  })

  it('adds no trailing blank line when the reply lands at the end of the page', () => {
    const doc = 'a note\n\n@hermes what is this?'

    expect(replyInsertion(doc, '@hermes what is this?', 'reply').insert).toBe('\n\nreply')
  })

  it('falls back to the end of the page when the question was edited away', () => {
    const doc = 'the question is gone now'

    expect(replyInsertion(doc, '@hermes what is this?', 'reply').from).toBe(doc.length)
  })
})
