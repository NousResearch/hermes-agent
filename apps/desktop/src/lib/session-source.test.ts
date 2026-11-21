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

// LINE ships as a gateway *plugin* (plugins/platforms/line) rather than a
// built-in adapter, so it was missed when MESSAGING_SESSION_SOURCE_IDS was
// first written. Without the entry, refreshMessagingSessions() drops LINE rows
// by the same isMessagingSource() filter — LINE conversations then have no
// sidebar section and the platform has no label or icon. Same contract as the
// Photon guard above; these asserts fail loudly if the entry is removed.
describe('line messaging source registration', () => {
  it('treats line as a messaging source (own sidebar section)', () => {
    expect(isMessagingSource('line')).toBe(true)
  })

  it('is case/space insensitive on the source id', () => {
    expect(isMessagingSource('LINE')).toBe(true)
    expect(isMessagingSource('  line ')).toBe(true)
  })

  it('labels the source as LINE', () => {
    expect(sessionSourceLabel('line')).toBe('LINE')
  })
})
