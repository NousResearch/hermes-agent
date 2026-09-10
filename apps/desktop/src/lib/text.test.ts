import { describe, expect, it } from 'vitest'
import { asText, includesQuery, prettyName, normalize, capitalize, firstStringField } from '@/lib/text'

describe('asText', () => {
  it('passes through strings', () => {
    expect(asText('hello')).toBe('hello')
  })
  it('converts numbers', () => {
    expect(asText(42)).toBe('42')
  })
  it('returns empty for null/undefined', () => {
    expect(asText(null)).toBe('')
    expect(asText(undefined)).toBe('')
  })
  it('stringifies objects', () => {
    expect(asText({a:1})).toBe('[object Object]')
  })
})

describe('includesQuery', () => {
  it('case-insensitive match', () => {
    expect(includesQuery('Hello World', 'world')).toBe(true)
  })
  it('no match returns false', () => {
    expect(includesQuery('Hello', 'xyz')).toBe(false)
  })
})

describe('prettyName', () => {
  it('replaces underscores and capitalizes', () => {
    expect(prettyName('hello_world')).toBe('Hello World')
  })
  it('single word', () => {
    expect(prettyName('test')).toBe('Test')
  })
})

describe('normalize', () => {
  it('trims and lowercases', () => {
    expect(normalize('  Hello World  ')).toBe('hello world')
  })
  it('handles non-strings', () => {
    expect(normalize(123)).toBe('123')
  })
})

describe('capitalize', () => {
  it('capitalizes first char', () => {
    expect(capitalize('hello')).toBe('Hello')
  })
  it('empty string stays empty', () => {
    expect(capitalize('')).toBe('')
  })
  it('already capitalized stays same', () => {
    expect(capitalize('Hello')).toBe('Hello')
  })
})

describe('firstStringField', () => {
  it('returns first non-empty string field', () => {
    const record = { a: '', b: 'found', c: 'also' }
    expect(firstStringField(record, ['a', 'b', 'c'])).toBe('found')
  })
  it('returns empty when no match', () => {
    expect(firstStringField({ a: '' }, ['a'])).toBe('')
  })
  it('skips non-string values', () => {
    expect(firstStringField({ a: 42, b: 'text' }, ['a', 'b'])).toBe('text')
  })
})
