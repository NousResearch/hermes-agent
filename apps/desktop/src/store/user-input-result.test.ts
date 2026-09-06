import { describe, expect, it } from 'vitest'

import {
  classifyUserInputResult,
  createUserInputSubmitLock
} from './user-input-result'

describe('native user-input acknowledgements', () => {
  it('keeps invalid answers retryable and does not clear the request', () => {
    expect(classifyUserInputResult({
      accepted: false,
      error: { code: 'invalid', message: 'Answer required: Pick one' },
      status: 'invalid'
    })).toMatchObject({
      clear: false,
      kind: 'invalid',
      retryable: true,
      status: 'invalid'
    })
  })

  it('clears only an explicitly terminal or gone request', () => {
    expect(classifyUserInputResult({ accepted: false, status: 'expired' })).toMatchObject({
      clear: true,
      kind: 'terminal',
      status: 'expired'
    })
    expect(classifyUserInputResult({ accepted: false, error: { code: 'not_found' } })).toMatchObject({
      clear: true,
      kind: 'not_found',
      status: 'not_found'
    })
  })

  it('does not treat an arbitrary HTTP success or malformed body as accepted', () => {
    expect(classifyUserInputResult({}, 200)).toMatchObject({
      clear: false,
      kind: 'malformed',
      retryable: true
    })
    expect(classifyUserInputResult({ accepted: true, status: 'answered', delivery: 'unknown' })).toMatchObject({
      clear: false,
      kind: 'malformed',
      retryable: true
    })
  })

  it('does not allow overlapping submits for the same owned request', () => {
    const lock = createUserInputSubmitLock()

    expect(lock.acquire('session-1', 'request-1')).toBe(true)
    expect(lock.acquire('session-1', 'request-1')).toBe(false)
    expect(lock.acquire('session-1', 'request-2')).toBe(true)
    lock.release('session-1', 'request-1')
    expect(lock.acquire('session-1', 'request-1')).toBe(true)
  })
})
