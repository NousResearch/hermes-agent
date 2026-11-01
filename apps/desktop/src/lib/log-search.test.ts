import { describe, expect, it } from 'vitest'

import { findLogSearchHits, logLineSegments, logLineSeverity } from './log-search'

describe('findLogSearchHits', () => {
  it('returns offsets that slice the original line back to the query, case-insensitively', () => {
    // U+0130 lowercases to two code units; offsets taken from a lowercased copy
    // would land one character early on everything after it.
    const lines = ['İstanbul Docker', 'docker DOCKER', 'nothing']
    const hits = findLogSearchHits(lines, '  docker ')

    expect(hits).toHaveLength(3)

    for (const hit of hits) {
      expect(lines[hit.line].slice(hit.start, hit.end).toLowerCase()).toBe('docker')
    }
  })

  it('treats the query as a literal, not a pattern', () => {
    expect(findLogSearchHits(['a.b', 'axb'], 'a.b')).toEqual([{ end: 3, line: 0, start: 0 }])
  })
})

describe('logLineSegments', () => {
  it('reassembles into the original line with hits marked by their global index', () => {
    const line = 'x Docker y docker'
    const hits = findLogSearchHits([line], 'docker').map((hit, index) => ({ hit, index: index + 5 }))
    const segments = logLineSegments(line, hits)

    expect(segments.map(segment => segment.text).join('')).toBe(line)
    expect(segments.filter(segment => segment.hit !== null).map(segment => segment.hit)).toEqual([5, 6])
  })
})

describe('logLineSeverity', () => {
  it('reads the level column, not a level word inside the message', () => {
    expect(logLineSeverity('2026-09-23 21:15:07,123 WARNING gateway.run: retrying')).toBe('warning')
    expect(logLineSeverity('2026-09-23 21:15:07,123 INFO tools: ERROR count=0')).toBeNull()
    expect(logLineSeverity('ERROR without a timestamp')).toBeNull()
  })
})
