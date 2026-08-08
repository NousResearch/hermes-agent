import { describe, expect, it } from 'vitest'

import { sessionShowsRunningArc } from './session-row-state'

describe('session row running appearance', () => {
  it('keeps the running arc when an authoritative turn becomes quiet', () => {
    expect(sessionShowsRunningArc({ isWorking: true, needsInput: false })).toBe(true)
  })

  it('uses the needs-input treatment instead of the running arc', () => {
    expect(sessionShowsRunningArc({ isWorking: true, needsInput: true })).toBe(false)
  })

  it('shows no arc for a session that is not running', () => {
    expect(sessionShowsRunningArc({ isWorking: false, needsInput: false })).toBe(false)
  })
})
