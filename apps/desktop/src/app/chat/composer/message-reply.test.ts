import { describe, expect, it } from 'vitest'

import { insertMessageReply, quoteMessageForReply } from './message-reply'

describe('message reply composer insertion', () => {
  it('inserts every line, including blank lines, as one block reply', () => {
    const inserts: Array<{ options: { mode: 'block'; target: 'main' }; text: string }> = []

    const inserted = insertMessageReply('First line\n\nSecond line', (text, options) => {
      inserts.push({ options, text })
    })

    expect(inserted).toBe(true)
    expect(inserts).toEqual([
      {
        options: { mode: 'block', target: 'main' },
        text: '> First line\n> \n> Second line'
      }
    ])
    expect(inserts[0]?.text.split('\n').every(line => line.startsWith('> '))).toBe(true)
  })

  it('ignores an empty message', () => {
    let inserted = false

    expect(quoteMessageForReply('  \n  ')).toBe('')
    expect(
      insertMessageReply('  \n  ', () => {
        inserted = true
      })
    ).toBe(false)
    expect(inserted).toBe(false)
  })
})
