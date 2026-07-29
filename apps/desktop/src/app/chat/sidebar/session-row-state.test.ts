import { describe, expect, it } from 'vitest'

import { sessionDotState, sessionShowsRunningArc } from './session-row-state'

describe('session row running appearance', () => {
  it('keeps the running arc when an authoritative turn becomes quiet', () => {
    expect(sessionShowsRunningArc({ isWorking: true, needsInput: false })).toBe(true)
    expect(
      sessionDotState({
        hasBackground: false,
        isStalled: true,
        isUnread: false,
        isWorking: true,
        needsInput: false
      })
    ).toBe('stalled')
  })

  it('uses the needs-input treatment instead of the running arc', () => {
    expect(sessionShowsRunningArc({ isWorking: true, needsInput: true })).toBe(false)
    expect(
      sessionDotState({
        hasBackground: true,
        isStalled: true,
        isUnread: true,
        isWorking: true,
        needsInput: true
      })
    ).toBe('needs-input')
  })

  it('keeps background and unread states below active-turn states', () => {
    expect(
      sessionDotState({
        hasBackground: true,
        isStalled: false,
        isUnread: true,
        isWorking: false,
        needsInput: false
      })
    ).toBe('background')
  })
})
