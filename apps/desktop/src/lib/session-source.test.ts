import { describe, expect, it } from 'vitest'

import { isMessagingSource, MESSAGING_SESSION_SOURCE_IDS, sessionSourceLabel } from './session-source'

describe('Kindle session source', () => {
  it('is surfaced as a messaging-platform sidebar section', () => {
    expect(MESSAGING_SESSION_SOURCE_IDS).toContain('kindle')
    expect(isMessagingSource('kindle')).toBe(true)
    expect(sessionSourceLabel('kindle')).toBe('Kindle')
  })
})
