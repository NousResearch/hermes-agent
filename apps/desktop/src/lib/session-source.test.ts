import { describe, expect, it } from 'vitest'

import {
  isMessagingSource,
  MESSAGING_SESSION_SOURCE_IDS,
  sessionSourceLabel,
  sessionSourceSearchTerms,
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

describe('session-source registry', () => {
  it('groups Hermes Browser Extension sessions as their own sidebar source', () => {
    expect(sessionSourceLabel('hermes_browser')).toBe('Hermes Browser Extension')
    expect(MESSAGING_SESSION_SOURCE_IDS).toContain('hermes_browser')
    expect(isMessagingSource('hermes_browser')).toBe(true)
    expect(sessionSourceSearchTerms('hermes_browser')).toContain('Hermes Browser Extension')
  })

  it('groups Hermes Web sessions as their own sidebar source', () => {
    expect(sessionSourceLabel('hermes_web')).toBe('Hermes Web')
    expect(MESSAGING_SESSION_SOURCE_IDS).toContain('hermes_web')
    expect(isMessagingSource('hermes_web')).toBe(true)
    expect(sessionSourceSearchTerms('hermes_web')).toContain('Hermes Web')
  })
})
