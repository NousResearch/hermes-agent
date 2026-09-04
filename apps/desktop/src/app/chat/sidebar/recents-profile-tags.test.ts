import { describe, expect, it } from 'vitest'

import { showProfileTagsInRecents } from './recents-profile-tags'

describe('showProfileTagsInRecents', () => {
  it.each([
    { groupedByProfile: false, showAllProfiles: true, visible: true },
    { groupedByProfile: true, showAllProfiles: true, visible: false },
    { groupedByProfile: false, showAllProfiles: false, visible: false }
  ])(
    'returns $visible when allProfiles=$showAllProfiles and groupedByProfile=$groupedByProfile',
    ({ groupedByProfile, showAllProfiles, visible }) => {
      expect(showProfileTagsInRecents(showAllProfiles, groupedByProfile)).toBe(visible)
    }
  )
})
