import { describe, expect, it, vi } from 'vitest'

import type { SessionInfo, SessionSearchResult } from '@/types/hermes'

import {
  mergeSessionSearchResults,
  searchResultToSession,
  sessionMatchesSearch,
  stripFtsMarkers
} from './session-search'

function makeSession(overrides: Partial<SessionInfo> = {}): SessionInfo {
  return {
    archived: false,
    cwd: '/home/user/projects/hermes-agent',
    ended_at: null,
    id: '20260603_090200_abcd12',
    input_tokens: 0,
    is_active: false,
    last_active: 1_000,
    message_count: 2,
    model: 'claude',
    output_tokens: 0,
    preview: 'Fix Desktop session search',
    source: 'cli',
    started_at: 1_000,
    title: 'Desktop Search Feature',
    tool_call_count: 0,
    ...overrides
  }
}

describe('sessionMatchesSearch', () => {
  it('matches loaded sessions by full and partial session id', () => {
    const session = makeSession()

    expect(sessionMatchesSearch(session, '20260603_090200_abcd12')).toBe(true)
    expect(sessionMatchesSearch(session, '090200')).toBe(true)
    expect(sessionMatchesSearch(session, 'ABCD12')).toBe(true)
  })

  it('matches projected compression sessions by lineage root id', () => {
    const session = makeSession({
      _lineage_root_id: '20260602_235959_root99',
      id: '20260603_010000_tip01'
    })

    expect(sessionMatchesSearch(session, 'root99')).toBe(true)
    expect(sessionMatchesSearch(session, '20260602')).toBe(true)
  })

  it('preserves title, preview, and workspace matching', () => {
    const session = makeSession()

    expect(sessionMatchesSearch(session, 'desktop search')).toBe(true)
    expect(sessionMatchesSearch(session, 'session search')).toBe(true)
    expect(sessionMatchesSearch(session, 'hermes-agent')).toBe(true)
  })

  it('matches sessions by git branch', () => {
    expect(sessionMatchesSearch(makeSession({ git_branch: 'feat/cool-thing' }), 'feat/cool-thing')).toBe(true)
    expect(sessionMatchesSearch(makeSession({ git_branch: 'feat/cool-thing' }), 'cool')).toBe(true)
    expect(sessionMatchesSearch(makeSession({ git_branch: 'main' }), 'main')).toBe(true)
  })

  it('matches sessions by source platform and aliases', () => {
    expect(sessionMatchesSearch(makeSession({ source: 'telegram' }), 'Telegram')).toBe(true)
    expect(sessionMatchesSearch(makeSession({ source: 'whatsapp' }), 'WhatsApp')).toBe(true)
    expect(sessionMatchesSearch(makeSession({ source: 'whatsapp' }), 'wa')).toBe(true)
    expect(sessionMatchesSearch(makeSession({ source: 'slack' }), 'slack')).toBe(true)
    expect(sessionMatchesSearch(makeSession({ source: 'bluebubbles' }), 'imessage')).toBe(true)
  })

  it('does not match unrelated queries', () => {
    expect(sessionMatchesSearch(makeSession(), 'totally-unrelated')).toBe(false)
  })
})

describe('stripFtsMarkers', () => {
  it('removes the sqlite snippet() highlight delimiters', () => {
    expect(stripFtsMarkers('fix >>>session<<< search in <<<Desktop>>>')).toBe('fix session search in Desktop')
  })

  it('leaves plain text and empty strings untouched', () => {
    expect(stripFtsMarkers('no markers here')).toBe('no markers here')
    expect(stripFtsMarkers('')).toBe('')
  })

  it('removes every marker pair, not just the first', () => {
    expect(stripFtsMarkers('>>>a<<< and >>>b<<< and >>>c<<<')).toBe('a and b and c')
  })
})

describe('searchResultToSession', () => {
  it("carries the server's title, archived flag, timestamps and lineage through", () => {
    const result: SessionSearchResult = {
      archived: true,
      last_active: 1_800_000_200,
      lineage_root: '20260602_235959_root99',
      model: 'claude',
      preview: 'please summarize artifacts',
      role: null,
      session_id: '20260603_010000_tip01',
      session_started: 1_800_000_000,
      snippet: 'please >>>summarize<<< artifacts',
      source: 'cli',
      started_at: 1_800_000_100,
      title: 'Recent content session'
    }

    const session = searchResultToSession(result)

    // Resume identity is the live compression tip, not the lineage root.
    expect(session.id).toBe('20260603_010000_tip01')
    expect(session._lineage_root_id).toBe('20260602_235959_root99')
    expect(session.archived).toBe(true)
    expect(session.title).toBe('Recent content session')
    // started_at wins over the older session_started; last_active passes through.
    expect(session.started_at).toBe(1_800_000_100)
    expect(session.last_active).toBe(1_800_000_200)
    expect(session.model).toBe('claude')
    expect(session.source).toBe('cli')
    // The preview comes from the marker-stripped snippet.
    expect(session.preview).toBe('please summarize artifacts')
  })

  it('falls back gracefully on a minimal payload', () => {
    vi.useFakeTimers()

    try {
      vi.setSystemTime(new Date(1_800_000_500 * 1000))

      const session = searchResultToSession({
        model: null,
        role: null,
        session_id: '20260604_120000_min01',
        session_started: 1_800_000_000,
        snippet: '>>>bare<<< hit',
        source: null
      })

      expect(session.archived).toBe(false)
      expect(session.title).toBeNull()
      // No started_at: session_started stands in for both clocks.
      expect(session.started_at).toBe(1_800_000_000)
      expect(session.last_active).toBe(1_800_000_000)
      expect(session.preview).toBe('bare hit')
      expect(session._lineage_root_id).toBeNull()
      expect(session.model).toBeNull()
      expect(session.source).toBeNull()
      expect(session.is_active).toBe(false)
      expect(session.message_count).toBe(0)
      expect(session.cwd).toBeNull()
    } finally {
      vi.useRealTimers()
    }
  })

  it('nulls the preview when both the stripped snippet and the stored preview are blank', () => {
    const session = searchResultToSession({
      model: null,
      role: null,
      session_id: 'x',
      session_started: null,
      snippet: '>>><<<',
      source: null
    })

    expect(session.preview).toBeNull()
  })

  it('falls back to the stored preview when the stripped snippet is blank', () => {
    const session = searchResultToSession({
      model: null,
      preview: 'the stored preview line',
      role: null,
      session_id: 'x',
      session_started: null,
      snippet: '>>><<<',
      source: null
    })

    expect(session.preview).toBe('the stored preview line')
  })

  it('prefers the stripped snippet over the stored preview', () => {
    const session = searchResultToSession({
      model: null,
      preview: 'stale stored line',
      role: null,
      session_id: 'x',
      session_started: null,
      snippet: '>>>fresh<<< snippet',
      source: null
    })

    expect(session.preview).toBe('fresh snippet')
  })
})

describe('mergeSessionSearchResults', () => {
  const local = makeSession({ id: 'local-1', preview: 'client match' })

  const serverHit = (id: string, overrides: Partial<SessionSearchResult> = {}): SessionSearchResult => ({
    model: null,
    role: null,
    session_id: id,
    session_started: 1_000,
    snippet: `hit for ${id}`,
    source: null,
    ...overrides
  })

  it('keeps local-only matches untouched', () => {
    const merged = mergeSessionSearchResults([local], [])

    expect(merged).toHaveLength(1)
    expect(merged[0]).toBe(local)
  })

  it('synthesizes rows for server-only matches', () => {
    const merged = mergeSessionSearchResults([], [serverHit('server-1', { title: 'Server title' })])

    expect(merged).toHaveLength(1)
    expect(merged[0].id).toBe('server-1')
    expect(merged[0].title).toBe('Server title')
    expect(merged[0].preview).toBe('hit for server-1')
  })

  it('lets the local row win when both sides match the same id', () => {
    const merged = mergeSessionSearchResults([local], [serverHit('local-1')])

    expect(merged).toHaveLength(1)
    expect(merged[0]).toBe(local)
  })

  it('prefers an already-loaded session over a synthesized row', () => {
    const loaded = makeSession({ id: 'server-1', preview: 'loaded row' })
    const merged = mergeSessionSearchResults([], [serverHit('server-1')], new Map([[loaded.id, loaded]]))

    expect(merged).toHaveLength(1)
    expect(merged[0]).toBe(loaded)
  })

  it('orders local matches before server matches', () => {
    const local2 = makeSession({ id: 'local-2' })
    const merged = mergeSessionSearchResults([local, local2], [serverHit('server-1'), serverHit('server-2')])

    expect(merged.map(s => s.id)).toEqual(['local-1', 'local-2', 'server-1', 'server-2'])
  })

  it('skips server hits without a usable session id', () => {
    const merged = mergeSessionSearchResults([], [serverHit(''), serverHit('server-3')])

    expect(merged.map(s => s.id)).toEqual(['server-3'])
  })

  it('skips a server hit whose lineage root is already listed as a local row', () => {
    // The store lists the durable root (pin id) while the backend hit the live
    // compression tip of that same conversation → one row, the local one.
    const local = makeSession({ id: 'root-1' })
    const merged = mergeSessionSearchResults([local], [serverHit('tip-new', { lineage_root: 'root-1' })])

    expect(merged).toHaveLength(1)
    expect(merged[0]).toBe(local)
    // Rows render with key={session.id} — ids must stay unique.
    expect(new Set(merged.map(row => row.id)).size).toBe(merged.length)
  })

  it('does not list a local tip row twice when the server hits the same lineage under another tip id', () => {
    // The reproduced blocker: the merge keyed `out` by session id but tested
    // lineage membership against it, so the tip-vs-root identity gap let the
    // loaded fallback row in a second time — the same conversation on screen
    // twice under one React key.
    const loaded = makeSession({ _lineage_root_id: 'root-1', id: 'tip-1', preview: 'loaded row' })

    const loadedById = new Map([
      [loaded.id, loaded],
      [loaded._lineage_root_id!, loaded]
    ])

    const merged = mergeSessionSearchResults([loaded], [serverHit('tip-2', { lineage_root: 'root-1' })], loadedById)

    expect(merged).toHaveLength(1)
    expect(merged[0]).toBe(loaded)
    expect(merged.map(row => row.id)).toEqual(['tip-1'])
    expect(new Set(merged.map(row => row.id)).size).toBe(merged.length)
  })

  it('collapses two server hits that share one unloaded compression lineage', () => {
    // tip-A and tip-B are the same conversation (one lineage, two ids); with no
    // loaded row the first hit synthesizes the row and the second must be
    // skipped, not synthesized again under a different id.
    const merged = mergeSessionSearchResults([], [
      serverHit('tip-A', { lineage_root: 'root-1' }),
      serverHit('tip-B', { lineage_root: 'root-1' })
    ])

    expect(merged).toHaveLength(1)
    expect(merged[0].id).toBe('tip-A')
    expect(new Set(merged.map(row => row.id)).size).toBe(merged.length)
  })

  it('lists a lineage-root-keyed loaded row once, under its own id, for a tip-keyed server hit', () => {
    // The backend hit the live tip; loadedById indexes the richer loaded row
    // under its durable lineage root → that row wins over synthesis and is
    // keyed by loaded.id so it matches the id the list renders.
    const loaded = makeSession({ _lineage_root_id: 'root-9', id: 'tip-9', preview: 'loaded row' })

    const loadedById = new Map([
      [loaded.id, loaded],
      [loaded._lineage_root_id!, loaded]
    ])

    const merged = mergeSessionSearchResults(
      [],
      [serverHit('tip-new', { lineage_root: 'root-9', title: 'Server title' })],
      loadedById
    )

    expect(merged).toHaveLength(1)
    expect(merged[0]).toBe(loaded)
    expect(merged[0].id).toBe('tip-9')
    expect(new Set(merged.map(row => row.id)).size).toBe(merged.length)
  })

  it('skips a root-keyed server hit following an unloaded tip hit of the same lineage', () => {
    // Tip-then-root order: the tip hit synthesizes the row and records the
    // lineage; the later hit keyed by that root is the same conversation and
    // must not become a second row. First hit wins, so the row keeps tip-2.
    const merged = mergeSessionSearchResults([], [
      serverHit('tip-2', { lineage_root: 'root-1' }),
      serverHit('root-1', { lineage_root: null })
    ])

    expect(merged).toHaveLength(1)
    expect(merged[0].id).toBe('tip-2')
    expect(new Set(merged.map(row => row.id)).size).toBe(merged.length)
  })

  // Contract, not a bug: two LOCAL rows sharing a lineage root are both kept.
  // The backend's listable filter excludes compression children, so
  // list_sessions_rich(project_compression_tips=True) emits one row per
  // lineage; and the store keys identity by (profile, id)/(profile, lineage)
  // because another profile is a DIFFERENT session that must survive dedupe
  // (#92454) — collapsing local rows on a bare root would reintroduce that.
  it('keeps two local rows that share a lineage root across different profiles', () => {
    const tipA = makeSession({ _lineage_root_id: 'root-1', id: 'tip-a', profile: 'alpha' })
    const tipB = makeSession({ _lineage_root_id: 'root-1', id: 'tip-b', profile: 'beta' })

    const merged = mergeSessionSearchResults([tipA, tipB], [])

    expect(merged).toHaveLength(2)
    expect(merged[0]).toBe(tipA)
    expect(merged[1]).toBe(tipB)
  })
})
