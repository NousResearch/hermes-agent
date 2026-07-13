import { describe, expect, it } from 'vitest'

import { shouldShowSessionSections } from './section-states'

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
