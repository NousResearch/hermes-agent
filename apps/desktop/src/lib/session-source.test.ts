import { describe, expect, it } from 'vitest'

import { isMessagingSource, sessionSourceLabel, sessionSourceSearchTerms } from './session-source'

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

// Regression guard for #136856: the bundled SimpleX platform plugin writes
// sessions with source='simplex' (the plugin dir is the canonical Platform
// value). Before this entry existed, the messaging fetch filtered those rows
// out and they were rendered in no sidebar section at all.
describe('simplex messaging source registration', () => {
  it('treats simplex as a messaging source (own sidebar section)', () => {
    expect(isMessagingSource('simplex')).toBe(true)
  })

  it('is case/space insensitive on the source id', () => {
    expect(isMessagingSource('SIMPLEX')).toBe(true)
    expect(isMessagingSource('  simplex ')).toBe(true)
  })

  it('labels the platform SimpleX instead of the id-derived Simplex', () => {
    expect(sessionSourceLabel('simplex')).toBe('SimpleX')
  })

  it('exposes the source id and label as search terms', () => {
    const terms = sessionSourceSearchTerms('simplex')
    expect(terms).toContain('simplex')
    expect(terms).toContain('SimpleX')
  })
})
