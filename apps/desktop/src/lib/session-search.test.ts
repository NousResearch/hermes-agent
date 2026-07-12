import { describe, expect, it } from 'vitest'

import type { SessionInfo } from '@/types/hermes'

import { rankTitleMatchesFirst, sessionChannelMatches, sessionMatchesSearch, sessionOriginContext, sessionPlatformMatches, sessionTitleMatches } from './session-search'

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

  it('matches channel/thread names from display_name, tokenized (typo-tolerant per token)', () => {
    const discordThread = makeSession({
      display_name: 'Daemonarchy / #voice-assitant / Desktop App',
      source: 'discord',
      title: 'Pin-Sync Bug Recovery'
    })

    // channel name token + platform token, neither in the title
    expect(sessionMatchesSearch(discordThread, 'voice discord')).toBe(true)
    // thread name
    expect(sessionMatchesSearch(discordThread, 'desktop app')).toBe(true)
    // the channel-name typo doesn't break per-token matching of other tokens
    expect(sessionMatchesSearch(discordThread, 'daemonarchy voice')).toBe(true)
    // every token must land SOMEWHERE — a miss on one token fails the match
    expect(sessionMatchesSearch(discordThread, 'voice zzzzz')).toBe(false)
  })

  it('requires all tokens of a multi-word query to match across fields', () => {
    const session = makeSession({ source: 'telegram', title: 'Grocery list' })

    expect(sessionMatchesSearch(session, 'grocery telegram')).toBe(true)
    expect(sessionMatchesSearch(session, 'grocery discord')).toBe(false)
  })
})

describe('sessionChannelMatches', () => {
  it('hits when any token appears in the display_name path', () => {
    const s = makeSession({ display_name: 'Daemonarchy / #voice-assitant / Desktop App' })

    expect(sessionChannelMatches(s, 'voice')).toBe(true)
    expect(sessionChannelMatches(s, 'desktop')).toBe(true)
    expect(sessionChannelMatches(s, 'nope')).toBe(false)
  })

  it('never hits without a display_name', () => {
    expect(sessionChannelMatches(makeSession({ display_name: null }), 'voice')).toBe(false)
    expect(sessionChannelMatches(makeSession(), 'voice')).toBe(false)
  })
})

describe('sessionPlatformMatches', () => {
  it('hits on exact platform names and aliases only', () => {
    expect(sessionPlatformMatches(makeSession({ source: 'discord' }), 'discord')).toBe(true)
    expect(sessionPlatformMatches(makeSession({ source: 'telegram' }), 'tg')).toBe(true)
    expect(sessionPlatformMatches(makeSession({ source: 'discord' }), 'disc')).toBe(false)
  })
})

describe('sessionTitleMatches', () => {
  it('matches only the assigned title, case-insensitively', () => {
    const session = makeSession()

    expect(sessionTitleMatches(session, 'desktop search')).toBe(true)
    expect(sessionTitleMatches(session, 'DESKTOP')).toBe(true)
    // preview-only text is not a title match.
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

  it('ranks sessions matching MORE query tokens first (channel+platform beats single title token)', () => {
    const contentHit = makeSession({ id: 'content-1', preview: 'talked about voice stuff', title: 'Other' })
    const channelHit = makeSession({
      display_name: 'Daemonarchy / #voice-assitant',
      id: 'channel-1',
      source: 'discord',
      title: 'Firmware Build'
    })
    const platformOnly = makeSession({ id: 'platform-1', source: 'discord', title: 'Unrelated' })
    const titleHit = makeSession({ id: 'title-1', title: 'Voice pipeline design' })

    const ranked = rankTitleMatchesFirst([contentHit, channelHit, platformOnly, titleHit], 'voice discord')

    // channel-1 matches BOTH tokens (channel + platform) → first;
    // then single-token hits by tier: title(3) > platform(5); content last.
    expect(ranked.map(s => s.id)).toEqual(['channel-1', 'title-1', 'platform-1', 'content-1'])
  })

  it('does not let junk title-substring hits bury full channel-path matches ("desktop app" regression)', () => {
    // 22 cron sessions whose titles contain "app" (skill-patch-APPlier)
    const cronJunk = Array.from({ length: 5 }, (_, i) =>
      makeSession({ id: `cron-${i}`, source: 'cron', title: `skill-patch-applier · Jul 1${i}` })
    )
    const threadHit = makeSession({
      display_name: 'Daemonarchy / #voice-assitant / Desktop App',
      id: 'thread-1',
      source: 'discord',
      title: 'Pin-Sync Bug Recovery'
    })

    const ranked = rankTitleMatchesFirst([...cronJunk, threadHit], 'desktop app')

    // Both tokens hit the thread's channel path; cron titles only match "app".
    expect(ranked[0].id).toBe('thread-1')
  })

  it('whole-phrase title hits still beat multi-token channel hits', () => {
    const channelHit = makeSession({
      display_name: 'Daemonarchy / #voice-assitant / Desktop App',
      id: 'channel-1',
      source: 'discord',
      title: 'Other'
    })
    const phraseTitle = makeSession({ id: 'title-1', title: 'Troubleshooting Desktop App Timeout' })

    const ranked = rankTitleMatchesFirst([channelHit, phraseTitle], 'desktop app')

    // Both match 2 tokens; the whole-phrase title (tier 2) beats channel (tier 4).
    expect(ranked.map(s => s.id)).toEqual(['title-1', 'channel-1'])
  })

  it('returns input unchanged for an empty query', () => {
    const sessions = [makeSession({ id: 'a' }), makeSession({ id: 'b' })]

    expect(rankTitleMatchesFirst(sessions, '')).toEqual(sessions)
  })
})

describe('sessionOriginContext', () => {
  it('formats Discord channel/thread paths as Platform: channel: thread', () => {
    const s = makeSession({
      display_name: 'Daemonarchy / #voice-assitant / Desktop App',
      source: 'discord'
    })

    expect(sessionOriginContext(s)).toBe('Discord: voice-assitant: Desktop App')
  })

  it('formats a channel without a thread', () => {
    const s = makeSession({ display_name: 'Daemonarchy / #voice-assitant', source: 'discord' })

    expect(sessionOriginContext(s)).toBe('Discord: voice-assitant')
  })

  it('keeps single-segment display names (no guild to drop)', () => {
    const s = makeSession({ display_name: 'Home', source: 'telegram' })

    expect(sessionOriginContext(s)).toBe('Telegram: Home')
  })

  it('reduces local surfaces to the platform label alone', () => {
    expect(sessionOriginContext(makeSession({ display_name: null, source: 'tui' }))).toBe('TUI')
    expect(sessionOriginContext(makeSession({ display_name: null, source: 'cli' }))).toBe('CLI')
    expect(sessionOriginContext(makeSession({ display_name: null, source: 'desktop' }))).toBe('Desktop')
  })

  it('returns null when the source is unknown/empty', () => {
    expect(sessionOriginContext(makeSession({ display_name: null, source: null }))).toBeNull()
  })
})
