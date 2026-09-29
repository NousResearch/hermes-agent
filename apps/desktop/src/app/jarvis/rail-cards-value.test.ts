import { describe, expect, it } from 'vitest'

import type { VaultNoteNode } from '@/api/vault'

import { captureTitle, recentNotes } from './rail-memory-card'
import { topSources } from './rail-news-card'

const note = (id: string, timestamp: number): VaultNoteNode => ({
  excerpt: '',
  folder: '',
  id,
  label: id,
  links: 0,
  size: 0,
  tags: [],
  timestamp
})

describe('rail news sources', () => {
  it('offers the feeds with the most headlines first, capped', () => {
    const items = ['B', 'A', 'A', 'C', 'A', 'B', 'D', 'E'].map(source => ({ source }))

    expect(topSources(items, 3)).toEqual(['A', 'B', 'C'])
  })
})

describe('rail memory card', () => {
  it('lists the most recently changed notes, newest first', () => {
    expect(recentNotes([note('old', 1), note('new', 9), note('mid', 5)], 2).map(n => n.id)).toEqual(['new', 'mid'])
  })

  it('titles a captured thought with its first line, so it stays a valid, findable note name', () => {
    // Local wall-clock time (not UTC), and no colon a file name could not keep.
    const at = new Date(2026, 8, 30, 8, 15)
    const title = captureTitle('  Zadzwonić do Anny\ndruga linia', at)

    expect(title).toBe('2026-09-30 0815 Zadzwonić do Anny')
    expect(title).not.toContain(':')
    expect(captureTitle('   ', at)).toBe('2026-09-30 0815')
  })
})
