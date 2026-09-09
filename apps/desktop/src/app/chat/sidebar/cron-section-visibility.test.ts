import { describe, expect, it } from 'vitest'

import { shouldShowCronSection } from './cron-section-visibility'

describe('shouldShowCronSection', () => {
  it('shows with jobs and no active search', () => {
    expect(shouldShowCronSection({ cronJobsCount: 1, trimmedQuery: '' })).toBe(true)
  })

  it('hides when there are no jobs', () => {
    expect(shouldShowCronSection({ cronJobsCount: 0, trimmedQuery: '' })).toBe(false)
  })

  it('hides while a session search is active, same as the other non-search sections', () => {
    expect(shouldShowCronSection({ cronJobsCount: 5, trimmedQuery: 'foo' })).toBe(false)
  })

  // The contract that regressed: cron jobs aren't project-scoped, so unlike
  // agent sessions and messaging threads, Projects view has no equivalent
  // place to show them — the section must not depend on grouping mode.
  it('does not depend on worktree/Projects grouping', () => {
    expect(shouldShowCronSection({ cronJobsCount: 3, trimmedQuery: '' })).toBe(true)
  })
})
