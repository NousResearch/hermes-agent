import { expect, it } from 'vitest'

import { matchesSessionTags } from '@/lib/session-tags'
import type { SessionInfo, SessionSearchResult } from '@/types/hermes'

import { mergeSessionSearchResults } from './search-results'

it('ANDs shared filters with local search and server-only full-text hits after merging', () => {
  const local = [
    { id: 'local', title: 'needle', tags: ['work'] },
    { id: 'unrelated', title: 'other', tags: ['work'] }
  ] as SessionInfo[]

  const hits = [
    { session_id: 'server-tagged', snippet: 'body match', tags: ['work'] },
    { session_id: 'server-untagged', snippet: 'needle' }
  ] as SessionSearchResult[]

  expect(
    mergeSessionSearchResults('needle', local, hits, new Map(), s => matchesSessionTags(s, ['work'])).map(s => s.id)
  ).toEqual(['local', 'server-tagged'])
  expect(mergeSessionSearchResults('', local, hits, new Map(), () => true)).toEqual([])
  expect(mergeSessionSearchResults('needle', local, hits, new Map(), () => false)).toEqual([])
})
