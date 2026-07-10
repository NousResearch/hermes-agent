import { describe, expect, it } from 'vitest'

import type { SessionInfo } from '@/types/hermes'

import { rankTitleMatchesFirst, sessionMatchesSearch, sessionTitleMatches } from './session-search'

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

describe('sessionTitleMatches', () => {
  it('matches only the assigned title, case-insensitively', () => {
    const session = makeSession()

    expect(sessionTitleMatches(session, 'desktop search')).toBe(true)
    expect(sessionTitleMatches(session, 'DESKTOP')).toBe(true)
    // preview-only text is NOT a title match
    expect(sessionTitleMatches(session, 'Fix Desktop session')).toBe(false)
  })

  it('never matches untitled sessions or empty queries', () => {
    expect(sessionTitleMatches(makeSession({ title: null }), 'anything')).toBe(false)
    expect(sessionTitleMatches(makeSession(), '')).toBe(false)
    expect(sessionTitleMatches(makeSession(), '   ')).toBe(false)
  })
})

describe('rankTitleMatchesFirst', () => {
  it('floats title matches above preview/content matches, order preserved within groups', () => {
    const contentHit = makeSession({ id: 'content-1', preview: 'mentions deploy in text', title: 'Other' })
    const titleHitA = makeSession({ id: 'title-a', title: 'Deploy pipeline' })
    const contentHit2 = makeSession({ id: 'content-2', preview: 'deploy again', title: null })
    const titleHitB = makeSession({ id: 'title-b', title: 'Big deploy day' })

    const ranked = rankTitleMatchesFirst([contentHit, titleHitA, contentHit2, titleHitB], 'deploy')

    expect(ranked.map(s => s.id)).toEqual(['title-a', 'title-b', 'content-1', 'content-2'])
  })

  it('returns input unchanged for an empty query', () => {
    const sessions = [makeSession({ id: 'a' }), makeSession({ id: 'b' })]

    expect(rankTitleMatchesFirst(sessions, '')).toEqual(sessions)
  })
})
