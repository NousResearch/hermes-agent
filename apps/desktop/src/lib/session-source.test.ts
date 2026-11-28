import { describe, expect, it } from 'vitest'

import { isMessagingSource, sessionOriginName, sessionSourceSearchTerms } from './session-source'

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

// A messaging row's title never says WHO is on the other end (a DingTalk DM
// reads as its topic). The backend carries the sender's name in the row's
// origin payload; without this the sidebar shows several people's conversations
// with nothing to tell them apart.
describe('sessionOriginName', () => {
  it('reads the sender name from a DM origin payload', () => {
    const origin = JSON.stringify({ platform: 'dingtalk', chat_type: 'dm', user_name: '郑裕川' })

    expect(sessionOriginName({ origin_json: origin, source: 'dingtalk' })).toBe('郑裕川')
  })

  it('prefers the chat name for a group (its members are not the identity)', () => {
    const origin = JSON.stringify({ platform: 'dingtalk', chat_type: 'group', chat_name: '发布群', user_name: '郑裕川' })

    expect(sessionOriginName({ origin_json: origin, source: 'dingtalk' })).toBe('发布群')
  })

  it('returns null without a payload, without a name, or on malformed JSON', () => {
    expect(sessionOriginName({ origin_json: null, source: 'dingtalk' })).toBeNull()
    expect(sessionOriginName({ source: 'dingtalk' })).toBeNull()
    expect(
      sessionOriginName({ origin_json: JSON.stringify({ platform: 'dingtalk' }), source: 'dingtalk' })
    ).toBeNull()
    expect(sessionOriginName({ origin_json: 'not json', source: 'dingtalk' })).toBeNull()
  })

  it('returns null for a local session — only messaging platforms carry an origin name', () => {
    const origin = JSON.stringify({ platform: 'dingtalk', chat_type: 'dm', user_name: '郑裕川' })

    expect(sessionOriginName({ origin_json: origin, source: 'local' })).toBeNull()
  })
})
