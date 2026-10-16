import { describe, expect, it } from 'vitest'

import {
  configuredExcludeSources,
  isMessagingSource,
  recentsExcludedSources,
  recentsShowA2A,
  sessionSourceSearchTerms
} from './session-source'

// Regression guard for #46761 / PR #47395: Photon (iMessage) must keep its own
// sidebar section. refreshMessagingSessions() filters rows through
// isMessagingSource(), so this entry is the sole condition that keeps Photon
// sessions out of generic recents. A silent removal would regress the feature
// with no test failure — these asserts pin the contract.
describe('photon messaging source registration', () => {
  it('treats photon as a messaging source (own sidebar section)', () => {
    expect(isMessagingSource('photon')).toBe(true)
  })

  it('is case/space insensitive on the source id', () => {
    expect(isMessagingSource('PHOTON')).toBe(true)
    expect(isMessagingSource('  photon ')).toBe(true)
  })

  it('exposes the iMessage/messages search aliases so Photon sessions are findable', () => {
    const terms = sessionSourceSearchTerms('photon')
    expect(terms).toContain('imessage')
    expect(terms).toContain('messages')
  })

  it('does not flag local/CLI-ish sources as messaging (guard sanity)', () => {
    expect(isMessagingSource('cli')).toBe(false)
    expect(isMessagingSource(null)).toBe(false)
    expect(isMessagingSource(undefined)).toBe(false)
  })
})

// The recents slice hides machine sources by default: `a2a` peer dispatches are
// not conversations, and on a host that runs peers every inbound task mints a
// row (a 628-session store measured 36% A2A). `sessions.recents_show_a2a` is the
// user's way back in; `sessions.exclude_sources` adds extras on top. Search is
// never filtered by either, so a hidden session stays findable.
describe('recents source exclusions', () => {
  it('hides a2a by default and keeps the built-ins', () => {
    const excluded = recentsExcludedSources(['cron', 'kanban'], null)

    expect(excluded).toContain('a2a')
    expect(excluded).toContain('cron')
    expect(excluded).toContain('kanban')
  })

  it('adds sessions.exclude_sources on top, normalized and deduped', () => {
    const excluded = recentsExcludedSources(['cron'], {
      sessions: { exclude_sources: ['A2A', ' TUI ', 7, null, 'tui'] }
    })

    expect(excluded).toContain('a2a')
    expect(excluded).toContain('tui')
    expect(excluded.filter(id => id === 'tui')).toHaveLength(1)
  })

  it('returns the built-in array identity when the user opted a2a back in', () => {
    const builtIn = ['cron']

    expect(recentsExcludedSources(builtIn, { sessions: { recents_show_a2a: true } })).toBe(builtIn)
  })

  it('accepts only an explicit true as the opt-in', () => {
    expect(recentsShowA2A({ sessions: { recents_show_a2a: true } })).toBe(true)
    expect(recentsShowA2A({ sessions: { recents_show_a2a: 'true' } })).toBe(false)
    expect(recentsShowA2A({})).toBe(false)
    expect(recentsShowA2A(null)).toBe(false)
  })

  it('treats malformed exclude_sources values as nothing extra', () => {
    expect(configuredExcludeSources({ sessions: { exclude_sources: 'a2a' } })).toEqual([])
    expect(configuredExcludeSources({})).toEqual([])
    expect(configuredExcludeSources(null)).toEqual([])
  })
})
