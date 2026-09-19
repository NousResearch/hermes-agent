import { describe, expect, it } from 'vitest'

import { firstHttpUrl, linkLabel } from './ui'

describe('kanban linkify', () => {
  it('picks a markdown link before a bare URL', () => {
    expect(firstHttpUrl('[PROJ-12](https://example.com/browse/PROJ-12)\nhttps://example.org')).toBe(
      'https://example.com/browse/PROJ-12'
    )
  })

  it('picks a bare URL and strips trailing punctuation', () => {
    expect(firstHttpUrl('see https://example.com/browse/PROJ-12.')).toBe('https://example.com/browse/PROJ-12')
  })

  it('returns null when there is no URL', () => {
    expect(firstHttpUrl('PROJ-12 Move wallet api')).toBeNull()
  })

  it('labels a browse URL with the last path segment', () => {
    expect(linkLabel('https://example.com/browse/PROJ-12')).toBe('PROJ-12')
  })
})
