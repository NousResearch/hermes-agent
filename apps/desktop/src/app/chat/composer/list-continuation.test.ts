import { describe, expect, it } from 'vitest'

import { continuationDeleteLength, nextListMarker, parseListLineBefore } from './list-continuation'

describe('parseListLineBefore', () => {
  it('parses a numbered line', () => {
    expect(parseListLineBefore('1. first item')).toMatchObject({
      indent: '',
      marker: '1.',
      kind: 'numbered',
      number: 1,
      delimiter: '.',
      rest: 'first item'
    })
  })

  it('parses a paren-delimited numbered line', () => {
    expect(parseListLineBefore('3) third')).toMatchObject({
      marker: '3)',
      kind: 'numbered',
      number: 3,
      delimiter: ')',
      rest: 'third'
    })
  })

  it('parses a bulleted line and keeps the bullet', () => {
    expect(parseListLineBefore('- bullet')).toMatchObject({ marker: '-', kind: 'bulleted', rest: 'bullet' })
    expect(parseListLineBefore('* star')).toMatchObject({ marker: '*', rest: 'star' })
    expect(parseListLineBefore('+ plus')).toMatchObject({ marker: '+', rest: 'plus' })
  })

  it('keeps indentation', () => {
    expect(parseListLineBefore('  - nested')).toMatchObject({ indent: '  ', marker: '-', rest: 'nested' })
    expect(parseListLineBefore('\t1. tab')).toMatchObject({ indent: '\t' })
  })

  it('reads the last line only when the caret is on it', () => {
    expect(parseListLineBefore('prose\n2. list')).toMatchObject({ number: 2, rest: 'list' })
    expect(parseListLineBefore('2. list\nprose')).toBeNull()
  })

  it('parses a marker-only line with empty rest (end-of-list case)', () => {
    expect(parseListLineBefore('1.')).toMatchObject({ marker: '1.', rest: '' })
    expect(parseListLineBefore('-')).toMatchObject({ marker: '-', rest: '' })
  })

  it('rejects prose, a marker glued to text, and an empty editor', () => {
    expect(parseListLineBefore('hello world')).toBeNull()
    expect(parseListLineBefore('')).toBeNull()
    expect(parseListLineBefore('12abc')).toBeNull()
    expect(parseListLineBefore('-text')).toBeNull()
    expect(parseListLineBefore('a 1. b')).toBeNull()
  })
})

describe('nextListMarker', () => {
  it('increments a numbered marker', () => {
    const one = parseListLineBefore('1. first')
    const nine = parseListLineBefore('9) last')

    expect(one).not.toBeNull()
    expect(nine).not.toBeNull()
    expect(nextListMarker(one as NonNullable<typeof one>)).toBe('2. ')
    expect(nextListMarker(nine as NonNullable<typeof nine>)).toBe('10) ')
  })

  it('keeps a bulleted marker', () => {
    const dash = parseListLineBefore('- x')
    const star = parseListLineBefore('* x')

    expect(dash).not.toBeNull()
    expect(star).not.toBeNull()
    expect(nextListMarker(dash as NonNullable<typeof dash>)).toBe('- ')
    expect(nextListMarker(star as NonNullable<typeof star>)).toBe('* ')
  })
})

describe('continuationDeleteLength', () => {
  it('always returns 0 - we insert after the current line, never delete', () => {
    expect(continuationDeleteLength(parseListLineBefore('1. first item')!)).toBe(0)
    expect(continuationDeleteLength(parseListLineBefore('- x')!)).toBe(0)
    expect(continuationDeleteLength(parseListLineBefore('1.')!)).toBe(0)
  })
})
