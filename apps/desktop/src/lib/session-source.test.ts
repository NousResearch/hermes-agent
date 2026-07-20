import { describe, expect, it } from 'vitest'

import { isMessagingSource, sessionSourceLabel } from './session-source'

describe('session source metadata', () => {
  it('classifies LINE as a messaging source with its platform label', () => {
    expect(isMessagingSource('line')).toBe(true)
    expect(sessionSourceLabel('line')).toBe('LINE')
  })
})
