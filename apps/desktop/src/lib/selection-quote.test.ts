import { describe, expect, it } from 'vitest'

import {
  buildSelectionQuote,
  deriveSideChatTitle,
  quoteBlock,
  SELECTION_QUOTE_MAX_CHARS
} from './selection-quote'

/** Long enough that the cap must bite, with real line breaks to cut on. */
const overLong = Array.from({ length: 400 }, (_, index) => `line ${index} ${'x'.repeat(60)}`).join('\n')

describe('quoteBlock', () => {
  it('prefixes every line and keeps blank lines inside the blockquote', () => {
    expect(quoteBlock('one\n\ntwo').split('\n')).toEqual(['> one', '>', '> two'])
  })

  it('leaves interior indentation of code intact', () => {
    expect(quoteBlock('if (x) {\n  return y\n}')).toBe('> if (x) {\n>   return y\n> }')
  })
})

describe('buildSelectionQuote', () => {
  it('normalises line endings and outer whitespace without marking a short selection', () => {
    const quote = buildSelectionQuote('  a\r\n\r\nb  \n')

    expect(quote).toEqual({ text: '> a\n>\n> b', truncated: false })
  })

  it('returns nothing for a whitespace-only selection instead of an empty quote marker', () => {
    expect(buildSelectionQuote('   \n\n ')).toEqual({ text: '', truncated: false })
  })

  it('clips an over-budget selection at a line boundary and says so', () => {
    const quote = buildSelectionQuote(overLong)

    expect(quote.truncated).toBe(true)
    expect(quote.text.endsWith('> …(truncated)')).toBe(true)
    expect(quote.text.length).toBeLessThanOrEqual(SELECTION_QUOTE_MAX_CHARS)
    // Every emitted line is still a quoted line — the cut never leaves a
    // half-written line inside the blockquote.
    expect(quote.text.split('\n').every(line => line === '>' || line.startsWith('> '))).toBe(true)
  })

  it('keeps a selection that only just fits, so the cap is a cap and not a haircut', () => {
    const fitting = 'y'.repeat(200)
    const quote = buildSelectionQuote(fitting, 400)

    expect(quote.truncated).toBe(false)
    expect(quote.text).toBe(`> ${fitting}`)
  })

  it('still quotes a single unbroken line that exceeds the budget alone', () => {
    const quote = buildSelectionQuote('z'.repeat(900), 300)

    expect(quote.truncated).toBe(true)
    expect(quote.text.endsWith('> …(truncated)')).toBe(true)
    expect(quote.text.startsWith('> z')).toBe(true)
  })
})

describe('deriveSideChatTitle', () => {
  it('strips leading code punctuation so a code selection reads as a phrase', () => {
    expect(deriveSideChatTitle('```const cache = new Map()```')).toBe('About: const cache = new Map()')
  })

  it('collapses a multi-line selection into one tab-safe line', () => {
    const title = deriveSideChatTitle('the loader\n\nre-reads the file')

    expect(title).toBe('About: the loader re-reads the file')
    expect(title).not.toContain('\n')
  })

  it('truncates past the title budget on a word boundary', () => {
    const title = deriveSideChatTitle(`${'word '.repeat(40)}tail`)

    expect(title.startsWith('About: word')).toBe(true)
    expect(title.endsWith('…')).toBe(true)
    expect(title.length).toBeLessThan(60)
  })

  it('falls back to a usable label when nothing survives cleanup', () => {
    expect(deriveSideChatTitle('```\n  \n')).toBe('About: selection')
  })

  it('uses the caller’s translated prefix', () => {
    expect(deriveSideChatTitle('parseConfig', 'Über:')).toBe('Über: parseConfig')
  })
})
