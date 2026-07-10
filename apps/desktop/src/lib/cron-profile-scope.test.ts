import { describe, expect, it } from 'vitest'

import { ALL_PROFILES } from '@/store/profile'

import { cronJobsProfileForScope } from './cron-profile-scope'

describe('cronJobsProfileForScope', () => {
  it('keeps concrete profile cron lists scoped to that profile', () => {
    expect(cronJobsProfileForScope('default')).toBe('default')
    expect(cronJobsProfileForScope('coder')).toBe('coder')
  })

  it('maps the sidebar all-profiles scope to the backend aggregate profile', () => {
    expect(cronJobsProfileForScope(ALL_PROFILES)).toBe('all')
  })

  it('falls back to the primary profile for empty scope values', () => {
    expect(cronJobsProfileForScope(null)).toBe('default')
    expect(cronJobsProfileForScope('')).toBe('default')
  })
})
