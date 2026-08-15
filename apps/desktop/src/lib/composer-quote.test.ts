import { describe, expect, it } from 'vitest'

import { composerQuoteLabel, decodeComposerQuote, encodeComposerQuote, expandComposerQuotes } from './composer-quote'
import { deriveDraftTitle } from './draft-title'

const quoteRef = (label: string, body: string) => `@quote:\`${encodeComposerQuote({ body, label })}\``

describe('self-contained composer quote refs', () => {
  it('round-trips Unicode, emoji, and punctuation', () => {
    const payload = encodeComposerQuote({ body: '> 你好 👋!', label: '你好 👋!' })

    expect(decodeComposerQuote(payload)).toEqual({ body: '> 你好 👋!', label: '你好 👋!' })
    expect(composerQuoteLabel('`' + payload + '`')).toBe('你好 👋!')
  })

  it('leaves malformed or legacy quote refs visible', () => {
    expect(decodeComposerQuote('q1.not_base64%')).toBeNull()
    expect(expandComposerQuotes('@quote:`old label` response')).toBe('@quote:`old label` response')
  })

  it('survives a plain persisted draft round trip without a side store', () => {
    const draft = quoteRef('Alpha?', '> Alpha?') + ' A response'
    const restored = JSON.parse(JSON.stringify(draft)) as string

    expect(expandComposerQuotes(restored)).toBe('> Alpha?\n\nA response')
  })

  it('derives a readable tab title from the decoded quote label', () => {
    const draft = quoteRef('Earlier answer?', '> Earlier answer?') + ' Correction'

    expect(deriveDraftTitle(draft)).toBe('Earlier answer? Correction')
  })
})
