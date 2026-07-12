import { describe, expect, it } from 'vitest'

import {
  MESSAGING_SESSION_SOURCE_IDS,
  isMessagingSource,
  sessionSourceLabel,
  sessionSourceSearchTerms,
} from './session-source'

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
