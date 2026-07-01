import { describe, expect, it } from 'vitest'

import { PROJECT_PREVIEW_COUNT } from './model'
import { PROJECT_OVERVIEW_PREVIEW_LIMIT } from './workspace-groups'

describe('PROJECT_PREVIEW_COUNT', () => {
  it('uses the shared project overview preview limit', () => {
    expect(PROJECT_PREVIEW_COUNT).toBe(PROJECT_OVERVIEW_PREVIEW_LIMIT)
    expect(PROJECT_PREVIEW_COUNT).toBe(10)
  })
})
