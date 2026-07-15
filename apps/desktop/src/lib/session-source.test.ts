import { describe, expect, it } from 'vitest'

import {
  isMessagingSource,
  MESSAGING_SESSION_SOURCE_IDS,
  sessionOriginBadgeSource,
  sessionSourceLabel,
  sessionSourceSearchTerms
} from './session-source'

describe('session-source', () => {
  it('marks Photon iMessage sessions without moving them into a separate sidebar section', () => {
    expect(MESSAGING_SESSION_SOURCE_IDS).not.toContain('photon')
    expect(isMessagingSource('photon')).toBe(false)
    expect(sessionOriginBadgeSource('photon', null, null)).toBe('photon')
    expect(sessionSourceLabel('photon')).toBe('iMessage / Photon')
    expect(sessionSourceSearchTerms('photon')).toEqual(expect.arrayContaining(['photon', 'iMessage / Photon', 'imessage']))
  })
})
