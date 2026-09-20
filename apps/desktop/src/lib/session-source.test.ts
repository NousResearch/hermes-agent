import { describe, expect, it } from 'vitest'

import {
  isMessagingSource,
  MESSAGING_SESSION_SOURCE_IDS,
  sessionSourceLabel,
  sessionSourceSearchTerms
} from './session-source'

const PLUGIN_BACKED_MESSAGING_SOURCES = [
  { id: 'google_chat', label: 'Google Chat' },
  { id: 'irc', label: 'IRC' },
  { id: 'line', label: 'LINE' },
  { id: 'ntfy', label: 'ntfy' },
  { id: 'raft', label: 'Raft' },
  { id: 'simplex', label: 'SimpleX Chat' },
  { id: 'teams', label: 'Microsoft Teams' }
] as const

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

// These adapters persist sessions under their plugin id. The desktop's
// messaging refresh filters every row through isMessagingSource() before the
// sidebar can group it, so a missing registration leaves real conversations
// mixed into generic recents instead of giving the platform its own section.
describe('plugin-backed messaging source registrations (#79836)', () => {
  it.each(PLUGIN_BACKED_MESSAGING_SOURCES)('$id is admitted to the messaging sidebar', ({ id }) => {
    expect(isMessagingSource(id)).toBe(true)
    expect(MESSAGING_SESSION_SOURCE_IDS).toContain(id)
  })

  it.each(PLUGIN_BACKED_MESSAGING_SOURCES)('$id uses its product label', ({ id, label }) => {
    expect(sessionSourceLabel(id)).toBe(label)
  })
})
