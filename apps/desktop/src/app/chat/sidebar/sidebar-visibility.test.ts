import { describe, expect, it } from 'vitest'

import { shouldShowMessagingSections } from './sidebar-visibility'

describe('shouldShowMessagingSections', () => {
  it('keeps messaging sections visible while workspace grouping is active', () => {
    expect(shouldShowMessagingSections({ searchActive: false, workspaceGroupingActive: true })).toBe(true)
  })

  it('keeps messaging sections visible in the flat recents view', () => {
    expect(shouldShowMessagingSections({ searchActive: false, workspaceGroupingActive: false })).toBe(true)
  })

  it('hides messaging sections during search results', () => {
    expect(shouldShowMessagingSections({ searchActive: true, workspaceGroupingActive: true })).toBe(false)
  })
})
