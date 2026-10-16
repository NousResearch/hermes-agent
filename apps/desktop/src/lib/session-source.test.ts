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

// whatsapp_cloud (official Meta Cloud API adapter) is a distinct gateway
// Platform from the Baileys bridge (`whatsapp`) and writes its own session
// source. Without this entry its sessions fall into generic recents while
// Telegram's get a sidebar section.
describe('whatsapp_cloud messaging source registration', () => {
  it('treats whatsapp_cloud as a messaging source (own sidebar section)', () => {
    expect(isMessagingSource('whatsapp_cloud')).toBe(true)
  })

  it('labels it distinctly from the Baileys bridge and keeps it findable as WhatsApp', () => {
    expect(sessionSourceLabel('whatsapp_cloud')).toBe('WhatsApp Business')
    const terms = sessionSourceSearchTerms('whatsapp_cloud')
    expect(terms).toContain('whatsapp')
    expect(terms).toContain('wa')
  })
})
