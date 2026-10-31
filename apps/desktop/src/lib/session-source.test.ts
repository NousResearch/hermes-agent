import { describe, expect, it } from 'vitest'

import { LOCAL_SESSION_SOURCE_IDS, isMessagingSource, sessionSourceSearchTerms } from './session-source'

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

// Regression guard for #146814: the dashboard's web chat labels sessions started
// from a phone browser `source='mobile'`. Because the messaging slice is defined
// negatively (excludeSources = ['cron', ...LOCAL_SESSION_SOURCE_IDS]), a source
// missing from this list renders as a platform chat with no stored-session resume
// and no delete action — the row is dead once the mobile WebSocket is reaped.
describe('mobile dashboard sessions are a local surface', () => {
  it('keeps mobile out of the messaging slice', () => {
    expect(isMessagingSource('mobile')).toBe(false)
  })

  it('lists mobile with the local sources so recents keep it and messaging excludes it', () => {
    expect(LOCAL_SESSION_SOURCE_IDS).toContain('mobile')
  })
})
