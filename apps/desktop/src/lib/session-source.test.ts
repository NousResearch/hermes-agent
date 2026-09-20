import { describe, expect, it } from 'vitest'

import { SIDEBAR_EXCLUDED_SOURCES } from '@/app/session/hooks/use-session-list-actions'

import { isMessagingSource, LOCAL_SESSION_SOURCE_IDS, MESSAGING_SESSION_SOURCE_IDS, sessionSourceSearchTerms } from './session-source'

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

// The Hermes IDE scopes its sessions with source='ide' (wire tag chosen once).
// These asserts pin the two-sided contract: the IDE's rows stay out of the
// primary sidebar's recents (via SIDEBAR_EXCLUDED_SOURCES) and out of the
// messaging slice (via LOCAL_SESSION_SOURCE_IDS), while still being a real
// local platform with a label. A silent removal of either entry would leak IDE
// sessions into the main window with no other failing test.
describe('Hermes IDE session source registration', () => {
  it('is a local (non-messaging) source', () => {
    expect(LOCAL_SESSION_SOURCE_IDS).toContain('ide')
    expect(isMessagingSource('ide')).toBe(false)
  })

  it('is excluded from the primary sidebar recents slice', () => {
    expect(SIDEBAR_EXCLUDED_SOURCES).toContain('ide')
  })

  it('carries a platform label for badges and search', () => {
    expect(sessionSourceSearchTerms('ide')).toContain('ide')
  })
})
