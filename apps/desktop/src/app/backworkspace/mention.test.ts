import { describe, expect, it } from 'vitest'

import { mentionInsertion, mentionTokenAt } from './mention'

describe('mentionTokenAt', () => {
  it('reads the mention being typed at the caret', () => {
    const doc = 'ask @her'

    expect(mentionTokenAt(doc, doc.length)).toEqual({ from: 4, query: 'her', to: 8 })
  })

  it('opens on a bare @ and on the first character of a line', () => {
    expect(mentionTokenAt('@', 1)).toEqual({ from: 0, query: '', to: 1 })
    expect(mentionTokenAt('note\n@bot', 9)).toEqual({ from: 5, query: 'bot', to: 9 })
  })

  it('stays closed where a mention does not start a word or the caret moved past it', () => {
    // An email address is not a mention.
    expect(mentionTokenAt('write to me@example.com', 23)).toBeNull()
    // The caret is before the mention, not inside it.
    expect(mentionTokenAt('ask @bot now', 3)).toBeNull()
    // A space ends the token.
    expect(mentionTokenAt('ask @bot now', 12)).toBeNull()
  })
})

describe('mentionInsertion', () => {
  it('replaces the typed token and leaves the caret after the trailing space', () => {
    const doc = 'ask @her'
    const token = mentionTokenAt(doc, doc.length)!

    expect(mentionInsertion(token, '@hermes')).toEqual({
      changes: { from: 4, insert: '@hermes ', to: 8 },
      selection: { anchor: 12 }
    })
  })
})
