import { describe, expect, it } from 'vitest'

import { ALL_PROFILES } from '@/store/profile'

import { shouldIncludeMessagingSession, shouldShowSessionSections } from './section-states'

const emptySidebar = {
  hasCronJobs: false,
  hasMessaging: false,
  hasProjects: false,
  hasSessions: false,
  loadingSessions: false
}

describe('shouldShowSessionSections', () => {
  it('keeps messaging visible without normal sessions', () => {
    expect(shouldShowSessionSections({ ...emptySidebar, hasMessaging: true })).toBe(true)
  })

  it('keeps cron jobs visible without normal sessions', () => {
    expect(shouldShowSessionSections({ ...emptySidebar, hasCronJobs: true })).toBe(true)
  })

  it('uses the blank state only when every section is empty', () => {
    expect(shouldShowSessionSections(emptySidebar)).toBe(false)
  })
})

describe('shouldIncludeMessagingSession', () => {
  it('keeps rows when a persisted All Profiles scope falls back to one profile', () => {
    expect(shouldIncludeMessagingSession(ALL_PROFILES, 'default')).toBe(true)
  })

  it('filters rows outside a concrete profile scope', () => {
    expect(shouldIncludeMessagingSession('work', 'default')).toBe(false)
    expect(shouldIncludeMessagingSession('work', 'work')).toBe(true)
  })
})
