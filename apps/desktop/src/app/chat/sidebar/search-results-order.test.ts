// Regression: pasting a session's exact id into the sidebar search buried the
// id's own row under recency-sorted client matches. Cross-reference workflows
// open sessions with a pointer line quoting another session's id, and the
// sidebar preview is built from that first message — so those sessions match
// the query client-side forever, while the backend ranks exact id matches
// first and the sidebar dropped that ranking at the merge.
import { describe, expect, it } from 'vitest'

import { makeSessionInfo } from '@/test/session-info'
import type { SessionInfo, SessionSearchResult } from '@/types/hermes'

import { mergeSearchResults } from './index'

const TARGET_ID = '20260914_183005_3738d0'

const quotingSession = makeSessionInfo({
  id: '20260915_090000_quote01',
  last_active: 5_000,
  preview: `resume from ${TARGET_ID}`,
  started_at: 4_900,
  title: 'Handoff pointer'
})

const targetSession = makeSessionInfo({
  id: TARGET_ID,
  last_active: 1_000,
  started_at: 900,
  title: 'The actual target'
})

const idHit: SessionSearchResult = {
  last_active: 1_000,
  lineage_root: null,
  model: null,
  role: null,
  session_id: TARGET_ID,
  session_started: 900,
  snippet: `Session ID: ${TARGET_ID}`,
  source: null
}

const indexByAnyId = (sessions: SessionInfo[]) => new Map(sessions.map(session => [session.id, session]))

describe('mergeSearchResults', () => {
  it('keeps the exact-id server hit above newer client matches once the response lands', () => {
    // The quoting session is more recent and matches the pasted id through its
    // preview; the backend still ranks the id's own conversation first.
    const merged = mergeSearchResults(
      [quotingSession, targetSession],
      TARGET_ID,
      [idHit],
      indexByAnyId([targetSession]),
      false
    )

    expect(merged.map(session => session.id)).toEqual([TARGET_ID, quotingSession.id])
  })

  it('leads client matches while the server request is still in flight', () => {
    const merged = mergeSearchResults(
      [quotingSession, targetSession],
      TARGET_ID,
      [],
      indexByAnyId([quotingSession, targetSession]),
      true
    )

    // Recency order until the ranked server response arrives — typing keeps
    // its instant feedback.
    expect(merged.map(session => session.id)).toEqual([quotingSession.id, TARGET_ID])
  })

  it("drops the previous query's server hits while the next request is in flight", () => {
    // The previous query landed a server-only row for the target; the user
    // then kept typing. Until the new response arrives, only the new query's
    // client matches may render — a fast typist could otherwise click a
    // result the search box no longer matches.
    const merged = mergeSearchResults(
      [quotingSession, targetSession],
      'Handoff',
      [idHit],
      indexByAnyId([quotingSession, targetSession]),
      true
    )

    expect(merged.map(session => session.id)).toEqual([quotingSession.id])
  })

  it('keeps a loaded conversation at its server rank, not its recency slot', () => {
    const merged = mergeSearchResults(
      [quotingSession, targetSession],
      TARGET_ID,
      [idHit],
      indexByAnyId([quotingSession, targetSession]),
      false
    )

    // The loaded row object renders (richer than the mapped hit), but the
    // server's position decides where.
    expect(merged.map(session => session.id)).toEqual([TARGET_ID, quotingSession.id])
    expect(merged[0]).toBe(targetSession)
  })

  it('maps an unloaded server hit with the backend last_active, not its start time', () => {
    const merged = mergeSearchResults([], TARGET_ID, [idHit], new Map(), false)

    expect(merged).toHaveLength(1)
    expect(merged[0].started_at).toBe(900)
    expect(merged[0].last_active).toBe(1_000)
  })

  it('falls back to the start time when the hit carries no last_active', () => {
    const bareHit: SessionSearchResult = { ...idHit, last_active: null }

    const merged = mergeSearchResults([], TARGET_ID, [bareHit], new Map(), false)

    expect(merged[0].last_active).toBe(900)
  })
})
