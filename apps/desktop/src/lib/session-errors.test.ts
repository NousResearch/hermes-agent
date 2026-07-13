import { describe, expect, it } from 'vitest'

import { isGatewayTimeoutError, isSessionBusyError, isSessionNotFoundError } from './session-errors'

describe('shared session error classifiers', () => {
  it('classifies gateway session failures from Error and string values', () => {
    expect(isSessionNotFoundError(new Error('Session not found'))).toBe(true)
    expect(isSessionNotFoundError('session NOT FOUND')).toBe(true)
    expect(isGatewayTimeoutError(new Error('request timed out: prompt.submit'))).toBe(true)
    expect(isGatewayTimeoutError('REQUEST TIMED OUT: session.resume')).toBe(true)
    expect(isSessionBusyError(new Error('session busy'))).toBe(true)
    expect(isSessionBusyError('SESSION BUSY')).toBe(true)
  })

  it('rejects unrelated values', () => {
    expect(isSessionNotFoundError(new Error('other'))).toBe(false)
    expect(isGatewayTimeoutError(undefined)).toBe(false)
    expect(isSessionBusyError({ message: 'session busy' })).toBe(false)
  })
})
