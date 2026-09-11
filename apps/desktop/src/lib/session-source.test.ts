import { describe, expect, it } from 'vitest'

import {
  configuredExcludeSources,
  isMessagingSource,
  mergeExcludedSources,
  MESSAGING_SESSION_SOURCE_IDS,
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

  it('is registered in the messaging source id list', () => {
    expect(MESSAGING_SESSION_SOURCE_IDS).toContain('photon')
  })

  it('does not flag local/CLI-ish sources as messaging (guard sanity)', () => {
    expect(isMessagingSource('cli')).toBe(false)
    expect(isMessagingSource(null)).toBe(false)
    expect(isMessagingSource(undefined)).toBe(false)
  })
})

// `sessions.exclude_sources` is read out of the raw config record
// (`Record<string, unknown>`), merged on top of the built-in exclusions. These
// pin the two properties the sidebar depends on: the built-in entries can never
// be configured away, and a malformed/absent key degrades to "nothing extra
// excluded" instead of throwing inside the render path.
describe('configured source exclusions', () => {
  it('reads and normalizes the configured list', () => {
    expect(configuredExcludeSources({ sessions: { exclude_sources: ['A2A', ' cron '] } })).toEqual(['a2a', 'cron'])
  })

  it('degrades to an empty list for absent / malformed values', () => {
    expect(configuredExcludeSources(undefined)).toEqual([])
    expect(configuredExcludeSources({})).toEqual([])
    expect(configuredExcludeSources({ sessions: {} })).toEqual([])
    expect(configuredExcludeSources({ sessions: { exclude_sources: 'a2a' } })).toEqual([])
    expect(configuredExcludeSources({ sessions: { exclude_sources: [1, null, ''] } })).toEqual([])
  })

  it('merges configured sources on top of the built-in ones, deduped', () => {
    expect(mergeExcludedSources(['cron', 'kanban'], ['a2a', 'cron'])).toEqual(['cron', 'kanban', 'a2a'])
  })

  it('returns the built-in array identity when nothing is configured', () => {
    const builtIn = ['cron', 'kanban']
    expect(mergeExcludedSources(builtIn, [])).toBe(builtIn)
  })
})
