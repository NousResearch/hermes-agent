import { describe, expect, it } from 'vitest'

import { isSessionGoneError } from './use-session-actions'

describe('isSessionGoneError', () => {
  it('matches the electron-wrapped 404 the delete endpoint actually returns', () => {
    expect(
      isSessionGoneError(
        new Error('Error invoking remote method \'hermes:api\': Error: 404: {"detail":"Session not found"}')
      )
    ).toBe(true)
  })

  it('matches bare and case-varied forms', () => {
    expect(isSessionGoneError(new Error('Session not found'))).toBe(true)
    expect(isSessionGoneError('session NOT found')).toBe(true)
  })

  it('does not match unrelated delete failures (those must still roll back)', () => {
    expect(isSessionGoneError(new Error('Timed out connecting to Hermes backend after 15000ms'))).toBe(false)
    expect(isSessionGoneError(new Error('500: internal error'))).toBe(false)
    expect(isSessionGoneError(undefined)).toBe(false)
  })
})
