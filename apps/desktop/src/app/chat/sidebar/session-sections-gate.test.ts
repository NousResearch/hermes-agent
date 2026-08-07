import { describe, expect, it } from 'vitest'

import { showSessionSectionsGate } from './session-sections-gate'

const noSessions = {
  showSessionSkeletons: false,
  localSessionCount: 0,
  messagingSessionCount: 0,
  cronSessionCount: 0,
  projectCount: 0
}

describe('showSessionSectionsGate', () => {
  it('hides the session sections when there is truly nothing to show', () => {
    expect(showSessionSectionsGate(noSessions)).toBe(false)
  })

  it('shows sections while local recents are loading (skeleton state)', () => {
    expect(showSessionSectionsGate({ ...noSessions, showSessionSkeletons: true })).toBe(true)
  })

  it('shows sections for local (recents) sessions alone', () => {
    expect(showSessionSectionsGate({ ...noSessions, localSessionCount: 1 })).toBe(true)
  })

  it('shows sections for projects alone', () => {
    expect(showSessionSectionsGate({ ...noSessions, projectCount: 1 })).toBe(true)
  })

  // Regression for issue #77816: a messaging-only user (e.g. WeChat/Telegram
  // via the gateway) with zero local sessions and zero projects must still
  // see their platform sections instead of the blank "No sessions" state.
  it('shows sections for messaging-platform sessions alone, with zero local sessions or projects', () => {
    expect(showSessionSectionsGate({ ...noSessions, messagingSessionCount: 1 })).toBe(true)
  })

  // Same regression class, noted in the issue as deserving the same fix:
  // a cron-only profile must not be hidden behind the blank state either.
  it('shows sections for cron sessions alone, with zero local sessions or projects', () => {
    expect(showSessionSectionsGate({ ...noSessions, cronSessionCount: 1 })).toBe(true)
  })

  it('shows sections when any combination is non-empty', () => {
    expect(
      showSessionSectionsGate({
        showSessionSkeletons: false,
        localSessionCount: 0,
        messagingSessionCount: 2,
        cronSessionCount: 1,
        projectCount: 0
      })
    ).toBe(true)
  })
})
