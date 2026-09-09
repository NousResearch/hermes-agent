import { describe, expect, it } from 'vitest'

import { isSessionOwnershipRefusal } from './session-ownership-refusal'

describe('isSessionOwnershipRefusal', () => {
  it('matches the active-session exclusivity message', () => {
    expect(
      isSessionOwnershipRefusal(
        'Session 20260909_095312_6b93f5 already has a live owner (tui, pid 32977, lease age 22m). ' +
          'Its turn activity is unknown; an open lease does not mean a turn is running. ' +
          'Attach through a compatible owner, or close the session in its owning surface ' +
          'before resuming here. Do not delete a live owner\'s lease to force a takeover.'
      )
    ).toBe(true)
  })

  it('matches the machine reason code', () => {
    expect(isSessionOwnershipRefusal('hermes-refusal-reason: SESSION_NOT_OWNED')).toBe(true)
  })

  it('ignores unrelated turn failures', () => {
    expect(isSessionOwnershipRefusal('rate limit exceeded')).toBe(false)
    expect(isSessionOwnershipRefusal('')).toBe(false)
    expect(isSessionOwnershipRefusal(null)).toBe(false)
  })
})
