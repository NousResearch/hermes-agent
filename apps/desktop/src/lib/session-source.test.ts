import { describe, expect, it } from 'vitest'

import { isMessagingSource, sessionSourceLabel } from './session-source'

describe('CTRL Systems session source', () => {
  it('renders as its own messaging section with a product label', () => {
    expect(isMessagingSource('ctrl_systems')).toBe(true)
    expect(sessionSourceLabel('ctrl_systems')).toBe('CTRL Systems')
  })
})
