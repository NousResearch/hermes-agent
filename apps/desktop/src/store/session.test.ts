import { describe, expect, it } from 'vitest'

import type { SessionInfo } from '@/types/hermes'

import { getMissingPinnedSessionIds, resolvePinnedSessions, sessionPinId } from './session'

const session = (over: Partial<SessionInfo>): SessionInfo => ({
  archived: false,
  cwd: null,
  ended_at: null,
  id: 'live',
  input_tokens: 0,
  is_active: false,
  last_active: 0,
  message_count: 0,
  model: null,
  output_tokens: 0,
  preview: null,
  source: null,
  started_at: 0,
  title: null,
  tool_call_count: 0,
  ...over
})

describe('sessionPinId', () => {
  it('uses the live id when there is no compression lineage', () => {
    expect(sessionPinId(session({ id: 'abc' }))).toBe('abc')
  })

  it('uses the lineage root so a pin survives compression', () => {
    // After auto-compression the entry surfaces under a fresh tip id but keeps
    // the original root — pinning on the root keeps the pin stable.
    expect(sessionPinId(session({ id: 'tip', _lineage_root_id: 'root' }))).toBe('root')
  })
})

describe('getMissingPinnedSessionIds', () => {
  it('returns pinned ids that cannot resolve from loaded sessions or fallbacks after reload', () => {
    const loaded = session({ id: 'loaded', title: 'Loaded chat' })
    const fallback = session({ id: 'cached-search', title: 'Cached search result' })

    expect(getMissingPinnedSessionIds(['loaded', 'cached-search', 'hidden-after-reload'], [loaded], [fallback])).toEqual([
      'hidden-after-reload'
    ])
  })

  it('does not hydrate a lineage-root pin that already resolves from a loaded continuation tip', () => {
    const loadedTip = session({ id: 'tip', _lineage_root_id: 'root', title: 'Loaded continuation' })

    expect(getMissingPinnedSessionIds(['root'], [loadedTip])).toEqual([])
  })
})

describe('resolvePinnedSessions', () => {
  it('renders a pinned search result that is absent from the loaded session list', () => {
    const loaded = session({ id: 'loaded', title: 'Loaded chat' })
    const searchOnly = session({ id: 'search-only', title: 'desktop ui bug' })

    expect(resolvePinnedSessions(['search-only'], [loaded], [searchOnly])).toEqual([searchOnly])
  })

  it('prefers the loaded session over a search-result fallback for the same pin id', () => {
    const loaded = session({ id: 'loaded', title: 'Loaded chat' })
    const staleSearchResult = session({ id: 'loaded', title: 'Stale search title' })

    expect(resolvePinnedSessions(['loaded'], [loaded], [staleSearchResult])).toEqual([loaded])
  })

  it('resolves search-result fallbacks by lineage root pin id', () => {
    const searchOnly = session({ id: 'tip', _lineage_root_id: 'root', title: 'Compressed search result' })

    expect(resolvePinnedSessions(['root'], [], [searchOnly])).toEqual([searchOnly])
  })
})
