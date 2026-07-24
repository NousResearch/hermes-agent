import { describe, expect, it } from 'vitest'

import {
  handoffOriginSource,
  isMessagingSource,
  sessionSourceLabel,
  sessionSourceSearchTerms
} from './session-source'

describe('session source metadata', () => {
  it('treats Photon iMessage chats as messaging sessions', () => {
    expect(isMessagingSource('photon')).toBe(true)
    expect(sessionSourceLabel('photon')).toBe('iMessage')
    expect(sessionSourceSearchTerms('photon')).toEqual(expect.arrayContaining(['photon', 'iMessage', 'imessage']))
  })

  it('uses Photon as a handoff origin badge but ignores local handoffs', () => {
    expect(handoffOriginSource('completed', 'photon')).toBe('photon')
    expect(handoffOriginSource('completed', 'desktop')).toBeNull()
    expect(handoffOriginSource('pending', 'photon')).toBeNull()
  })
})
