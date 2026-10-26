import { describe, expect, it } from 'vitest'

import { expandComposerQuotes } from '@/lib/composer-quote'

import { insertMessageReply, quoteMessageForReply } from './message-reply'

describe('message reply composer insertion', () => {
  it('routes a reply to the composer scope that owns the message', () => {
    const inserts: Array<{ options: { mode: 'block'; target: string }; text: string }> = []

    expect(
      insertMessageReply('Tile answer', {
        insert: (text, options) => inserts.push({ options, text }),
        target: 'tile:session-123'
      })
    ).toBe(true)

    expect(inserts).toEqual([
      {
        options: expect.objectContaining({ mode: 'block', target: 'tile:session-123' }),
        text: expect.any(String)
      }
    ])
  })

  it('can route a native selection to the active composer', () => {
    const targets: string[] = []

    insertMessageReply('selected text', {
      insert: (_text, options) => targets.push(options.target),
      target: 'active'
    })

    expect(targets).toEqual(['active'])
  })

  it('keeps a compact quote chip in the draft and expands it at submit time', () => {
    const inserts: Array<{ options: { mode: 'block'; target: string }; text: string }> = []

    const inserted = insertMessageReply('First line\n\nSecond line', {
      insert: (text, options) => inserts.push({ options, text }),
      target: 'main'
    })

    expect(inserted).toBe(true)
    expect(inserts).toHaveLength(1)
    expect(inserts[0]?.options).toEqual({ mode: 'block', target: 'main' })
    expect(inserts[0]?.text).toMatch(/^@quote:/)
    expect(inserts[0]?.text).not.toContain('\n')
    expect(expandComposerQuotes(`${inserts[0]?.text}My response`)).toBe(
      '> First line\n> \n> Second line\n\nMy response'
    )
  })

  it('formats every quoted line, including blank lines, as one block reply', () => {
    const quoted = quoteMessageForReply('First line\n\nSecond line')

    expect(quoted).toBe('> First line\n> \n> Second line')
    expect(quoted.split('\n').every(line => line.startsWith('> '))).toBe(true)
  })

  it('keeps two same-prefix replies mapped to distinct quote bodies', () => {
    const inserts: string[] = []
    const sharedPrefix = 'x'.repeat(80)
    const insert = (text: string) => void inserts.push(text)

    insertMessageReply(`${sharedPrefix} first`, { insert })
    insertMessageReply(`${sharedPrefix} second`, { insert })

    expect(inserts).toHaveLength(2)
    expect(inserts[0]).toMatch(/^@quote:/)
    expect(inserts[1]).toMatch(/^@quote:/)
    expect(inserts[0]).not.toBe(inserts[1])
    expect(expandComposerQuotes(inserts[0]!)).toContain('first')
    expect(expandComposerQuotes(inserts[1]!)).toContain('second')
  })

  it('keeps each live draft self-contained when the same message is quoted twice', () => {
    const inserts: string[] = []
    const insert = (text: string) => void inserts.push(text)

    insertMessageReply('same message', { insert })
    insertMessageReply('same message', { insert })

    expect(expandComposerQuotes(inserts[0]!)).toBe('> same message')
    expect(expandComposerQuotes(inserts[1]!)).toBe('> same message')
  })

  it('leaves an unresolved prototype-shaped label visible instead of throwing', () => {
    expect(expandComposerQuotes('@quote:`constructor` still visible')).toBe('@quote:`constructor` still visible')
  })

  it.each(['Done?', 'Yes!', 'Fine.'])('expands a quoted label ending in punctuation: %s', label => {
    const inserts: string[] = []

    insertMessageReply(label, { insert: text => void inserts.push(text) })

    expect(expandComposerQuotes(inserts[0]! + ' Reply')).toBe('> ' + label + '\n\nReply')
  })

  it('ignores an empty message', () => {
    let inserted = false

    expect(quoteMessageForReply('  \n  ')).toBe('')
    expect(
      insertMessageReply('  \n  ', {
        insert: () => {
          inserted = true
        }
      })
    ).toBe(false)
    expect(inserted).toBe(false)
  })
})
