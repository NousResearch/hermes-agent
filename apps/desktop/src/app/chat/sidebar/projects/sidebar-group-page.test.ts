import { describe, expect, it } from 'vitest'

import { PROJECT_PREVIEW_COUNT, SIDEBAR_GROUP_PAGE } from './model'

describe('SIDEBAR_GROUP_PAGE', () => {
  it('uses a practical page size for already-loaded workspace sessions', () => {
    expect(SIDEBAR_GROUP_PAGE).toBeGreaterThanOrEqual(20)
  })

  it('does not disturb profile preview count', () => {
    expect(PROJECT_PREVIEW_COUNT).toBe(3)
  })
})
