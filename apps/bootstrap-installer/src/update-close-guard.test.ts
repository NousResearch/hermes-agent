import { describe, expect, it } from 'vitest'

import { shouldBlockUpdateClose } from './update-close-guard'

describe('update close guard', () => {
  it('blocks user close only while an update is running', () => {
    expect(shouldBlockUpdateClose('update', 'running')).toBe(true)
    expect(shouldBlockUpdateClose('update', 'completed')).toBe(false)
    expect(shouldBlockUpdateClose('update', 'failed')).toBe(false)
    expect(shouldBlockUpdateClose('install', 'running')).toBe(false)
  })
})
