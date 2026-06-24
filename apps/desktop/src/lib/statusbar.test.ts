import { describe, expect, it } from 'vitest'

import { profileScopedStatusLabel, profileScopedStatusTitle } from './statusbar'

describe('profile-scoped statusbar labels', () => {
  it('keeps default profile labels compact', () => {
    expect(profileScopedStatusLabel('Running', 'default')).toBe('Running')
    expect(profileScopedStatusLabel('Running', '')).toBe('Running')
    expect(profileScopedStatusLabel('Running', null)).toBe('Running')
  })

  it('includes non-default profile names', () => {
    expect(profileScopedStatusLabel('Running', 'job-hunting')).toBe('Running · job-hunting')
  })

  it('describes multi-session running state in titles', () => {
    expect(profileScopedStatusTitle('Current turn elapsed', 'job-hunting', 3)).toBe(
      'Current turn elapsed · profile job-hunting · 3 sessions running'
    )
  })

  it('omits profile and count noise when there is no ambiguity', () => {
    expect(profileScopedStatusTitle('Current turn elapsed', 'default', 1)).toBe('Current turn elapsed')
  })
})
