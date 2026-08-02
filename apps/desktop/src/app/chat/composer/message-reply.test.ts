import { afterEach, describe, expect, it } from 'vitest'

import { $composerQuotes, clearComposerQuotes, expandComposerQuotes } from '@/store/composer'

import { insertMessageReply, quoteMessageForReply } from './message-reply'

afterEach(() => clearComposerQuotes?.())

describe('message reply composer insertion', () => {
  it('keeps a compact quote chip in the draft and expands it at submit time', () => {
    const inserts: Array<{ options: { mode: 'block'; target: 'main' }; text: string }> = []

    const inserted = insertMessageReply('First line\n\nSecond line', (text, options) => {
      inserts.push({ options, text })
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

    insertMessageReply(`${sharedPrefix} first`, insert)
    insertMessageReply(`${sharedPrefix} second`, insert)

    expect(inserts).toHaveLength(2)
    expect(inserts[0]).toMatch(/^@quote:/)
    expect(inserts[1]).toMatch(/^@quote:/)
    expect(inserts[0]).not.toBe(inserts[1])
    expect(expandComposerQuotes(inserts[0]!)).toContain('first')
    expect(expandComposerQuotes(inserts[1]!)).toContain('second')
  })

  it('allocates a distinct key when the same message is quoted into two live drafts', () => {
    const inserts: string[] = []
    const insert = (text: string) => void inserts.push(text)

    insertMessageReply('same message', insert)
    insertMessageReply('same message', insert)

    // A successful send removes its consumed body. Sharing a key would make
    // the still-open quote in another mounted composer lose its referent.
    expect(inserts[0]).not.toBe(inserts[1])
  })

  it('leaves an unresolved prototype-shaped label visible instead of throwing', () => {
    expect(expandComposerQuotes('@quote:`constructor` still visible')).toBe('@quote:`constructor` still visible')
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
    expect($composerQuotes?.get?.() ?? {}).toEqual({})
  })
})
