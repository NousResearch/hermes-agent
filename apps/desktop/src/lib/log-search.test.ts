import { describe, expect, it } from 'vitest'

import { countLogSearchMatches, firstLogSearchMatchLine, splitLogSearchMatches } from './log-search'

describe('log search', () => {
  it('counts every case-insensitive match across lines', () => {
    expect(countLogSearchMatches(['Docker docker', 'nothing', 'DOCKER'], 'docker')).toBe(3)
  })

  it('returns zero for an empty query', () => {
    expect(countLogSearchMatches(['anything'], '   ')).toBe(0)
  })

  it('finds the first matching line', () => {
    expect(firstLogSearchMatchLine(['alpha', 'beta needle', 'needle again'], 'NEEDLE')).toBe(1)
    expect(firstLogSearchMatchLine(['alpha', 'beta'], 'needle')).toBe(-1)
  })

  it('splits every match while preserving original text', () => {
    expect(splitLogSearchMatches('Docker docker DOCKER', 'docker')).toEqual([
      { match: true, text: 'Docker' },
      { match: false, text: ' ' },
      { match: true, text: 'docker' },
      { match: false, text: ' ' },
      { match: true, text: 'DOCKER' }
    ])
  })
})
