import { describe, expect, it } from 'vitest'

import type { SessionSearchResult } from '@/types/hermes'

import { searchResultToSession } from './index'

const result = (archived?: boolean): SessionSearchResult => ({
  archived,
  lineage_root: 'root-1',
  model: 'test-model',
  role: null,
  session_id: 'session-1',
  session_started: 1_785_946_000,
  snippet: 'matched text',
  source: 'desktop'
})

describe('searchResultToSession', () => {
  it('preserves archived search results as archived', () => {
    expect(searchResultToSession(result(true)).archived).toBe(true)
  })

  it('keeps compatibility with older backends that omit archived', () => {
    expect(searchResultToSession(result()).archived).toBe(false)
  })
})
