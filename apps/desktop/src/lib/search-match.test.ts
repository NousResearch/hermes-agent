import { describe, expect, it } from 'vitest'

import {
  excerptAround,
  levenshtein,
  parseHighlightSegments,
  parseQuery,
  rankFields,
  rankItems,
  scoreLabeledItem,
  stripHighlightMarkers,
  wrapRanges
} from './search-match'

describe('parseQuery', () => {
  it('parses AND by whitespace', () => {
    expect(parseQuery('foo bar').alternatives).toEqual([['foo', 'bar']])
  })

  it('parses OR and pipe', () => {
    expect(parseQuery('foo OR bar').alternatives).toEqual([['foo'], ['bar']])
    expect(parseQuery('a | b').alternatives).toEqual([['a'], ['b']])
  })

  it('keeps quoted phrases', () => {
    expect(parseQuery('"exact phrase" other').alternatives).toEqual([['exact phrase', 'other']])
  })
})

describe('rankFields', () => {
  const fields = [
    { field: 'title' as const, value: 'Desktop session search' },
    { field: 'preview' as const, value: 'fuzzy match and highlight' },
    { field: 'cwd' as const, value: 'C:/Users/me/hermes-agent' }
  ]

  it('matches substring (infix) not only prefix', () => {
    const hit = rankFields(fields, 'sess')
    expect(hit).not.toBeNull()
    expect(hit!.matches[0].field).toBe('title')
    // "sess" is a word-prefix of "session"
    expect(['word-prefix', 'substring']).toContain(hit!.matches[0].kind)

    const mid = rankFields([{ field: 'title', value: 'xxfoobarxx' }], 'bar')
    expect(mid!.matches[0].kind).toBe('substring')
  })

  it('prefers title over preview', () => {
    const hit = rankFields(fields, 'search')
    expect(hit!.matches[0].field).toBe('title')
  })

  it('supports OR alternatives', () => {
    const hit = rankFields(fields, 'zzzz OR fuzzy')
    expect(hit).not.toBeNull()
    expect(hit!.matches.some(m => m.field === 'preview')).toBe(true)
  })

  it('fuzzy-matches small typos on long enough words', () => {
    const hit = rankFields([{ field: 'name', value: 'docker-compose' }], 'dokcer', { fuzzy: true })
    expect(hit).not.toBeNull()
    expect(hit!.matches[0].kind).toBe('fuzzy')
  })

  it('does not fuzzy short needles', () => {
    const hit = rankFields([{ field: 'name', value: 'cat' }], 'ca', { fuzzy: true })
    // "ca" is prefix of cat
    expect(hit!.matches[0].kind).toBe('prefix')
  })
})

describe('rankItems + scoreLabeledItem', () => {
  it('ranks and filters a list', () => {
    const items = [
      { name: 'alpha', description: 'first' },
      { name: 'beta-tools', description: 'second alpha mention' },
      { name: 'gamma', description: 'nope' }
    ]
    const hits = rankItems(
      items,
      i => [
        { field: 'name', value: i.name },
        { field: 'description', value: i.description }
      ],
      'alpha'
    )
    expect(hits.map(h => h.item.name)).toEqual(['alpha', 'beta-tools'])
    expect(hits[0].score).toBeGreaterThan(hits[1].score)
  })

  it('scores palette labels with keywords', () => {
    const s = scoreLabeledItem('Open settings', ['prefs', 'config'], 'prefs')
    expect(s).toBeGreaterThan(0)
    expect(scoreLabeledItem('Open settings', ['prefs'], 'zzzz')).toBe(0)
  })
})

describe('levenshtein', () => {
  it('bounds distance', () => {
    expect(levenshtein('kitten', 'sitten', 1)).toBe(1)
    expect(levenshtein('kitten', 'sittin', 1)).toBeNull()
  })
})

describe('highlight helpers', () => {
  it('wraps and strips ranges', () => {
    const wrapped = wrapRanges('hello world', [[6, 11]])
    expect(wrapped).toContain('[[m]]world[[/m]]')
    expect(stripHighlightMarkers(wrapped)).toBe('hello world')
  })

  it('parses FTS markers', () => {
    const segs = parseHighlightSegments('before >>>hit<<< after')
    expect(segs).toEqual([
      { text: 'before ', hit: false },
      { text: 'hit', hit: true },
      { text: ' after', hit: false }
    ])
  })

  it('excerpts around a hit', () => {
    const long = 'x'.repeat(40) + 'NEEDLE' + 'y'.repeat(40)
    const { text, ranges } = excerptAround(long, [[40, 46]], 5)
    expect(text).toContain('NEEDLE')
    expect(text.startsWith('…')).toBe(true)
    expect(ranges[0][1] - ranges[0][0]).toBe(6)
  })
})
