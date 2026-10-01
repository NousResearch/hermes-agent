import { describe, expect, it } from 'vitest'

import {
  captureQuickEntryRequest,
  type QuickEntryAcceptedPromptStatus,
  validateQuickEntryAcceptance
} from './quick-entry-submission'

const CORRELATION_ID = 'qe-123'
const STORED_SESSION_ID = 'stored-session-1'
const RUNTIME_SESSION_ID = 'runtime-session-1'
const AFTER_DISPATCH = { promptDispatched: true }
const BEFORE_DISPATCH = { promptDispatched: false }

function capturedRequest(): ReturnType<typeof captureQuickEntryRequest> {
  return captureQuickEntryRequest({
    correlationId: CORRELATION_ID,
    identity: {
      prompt: 'Ship the exact prompt',
      storedSessionId: STORED_SESSION_ID,
      target: STORED_SESSION_ID
    }
  })
}

function backendResult(status: QuickEntryAcceptedPromptStatus, overrides: Record<string, unknown> = {}) {
  return {
    correlationId: CORRELATION_ID,
    runtimeSessionId: RUNTIME_SESSION_ID,
    sessionId: STORED_SESSION_ID,
    status,
    ...overrides
  }
}

describe('captureQuickEntryRequest', () => {
  it('freezes the request identity before an await can observe caller mutations', () => {
    const original = {
      correlationId: CORRELATION_ID,
      identity: {
        prompt: 'original prompt',
        storedSessionId: STORED_SESSION_ID,
        target: 'current'
      }
    }
    const request = captureQuickEntryRequest(original)

    original.correlationId = 'qe-other'
    original.identity.prompt = 'changed prompt'
    original.identity.storedSessionId = 'other-stored-session'
    original.identity.target = 'other-target'

    expect(request).toEqual({
      correlationId: CORRELATION_ID,
      identity: {
        prompt: 'original prompt',
        storedSessionId: STORED_SESSION_ID,
        target: 'current'
      }
    })
    expect(Object.isFrozen(request)).toBe(true)
    expect(Object.isFrozen(request.identity)).toBe(true)
  })
})

describe('validateQuickEntryAcceptance', () => {
  it.each(['streaming', 'queued', 'steered', 'redirected'] as const)(
    'accepts a correlated prompt result with status %s',
    status => {
      const outcome = validateQuickEntryAcceptance(capturedRequest(), backendResult(status), AFTER_DISPATCH)

      expect(outcome).toEqual({
        acceptedIdentity: {
          runtimeSessionId: RUNTIME_SESSION_ID,
          storedSessionId: STORED_SESSION_ID
        },
        correlationId: CORRELATION_ID,
        evidence: {
          observed: true,
          status
        },
        ok: true,
        requestedIdentity: {
          prompt: 'Ship the exact prompt',
          storedSessionId: STORED_SESSION_ID,
          target: STORED_SESSION_ID
        },
        status: 'accepted'
      })
    }
  )

  it('rejects an empty object as backend evidence', () => {
    expect(validateQuickEntryAcceptance(capturedRequest(), {}, AFTER_DISPATCH)).toMatchObject({
      ok: false,
      reason: 'invalid-backend-result',
      retryable: false,
      status: 'unknown'
    })
  })

  it('rejects a request whose identity is missing', () => {
    expect(validateQuickEntryAcceptance({ correlationId: CORRELATION_ID }, {}, AFTER_DISPATCH)).toMatchObject({
      ok: false,
      reason: 'invalid-request',
      retryable: false,
      status: 'unknown'
    })
  })

  it('rejects a backend result that omits correlation', () => {
    expect(
      validateQuickEntryAcceptance(capturedRequest(), backendResult('streaming', { correlationId: '' }), AFTER_DISPATCH)
    ).toMatchObject({
      ok: false,
      reason: 'invalid-backend-result',
      retryable: false,
      status: 'unknown'
    })
  })

  it('rejects a malformed backend response', () => {
    expect(validateQuickEntryAcceptance(capturedRequest(), { status: 'streaming' }, AFTER_DISPATCH)).toMatchObject({
      ok: false,
      reason: 'invalid-backend-result',
      retryable: false,
      status: 'unknown'
    })
  })

  it('rejects a non-prompt backend status', () => {
    expect(
      validateQuickEntryAcceptance(capturedRequest(), backendResult('completed' as never), AFTER_DISPATCH)
    ).toMatchObject({
      ok: false,
      reason: 'unsupported-backend-status',
      retryable: false,
      status: 'unknown'
    })
  })

  it('rejects acceptance for the wrong stored identity', () => {
    expect(
      validateQuickEntryAcceptance(
        capturedRequest(),
        backendResult('streaming', { sessionId: 'other-stored-session' }),
        AFTER_DISPATCH
      )
    ).toMatchObject({
      ok: false,
      reason: 'stored-identity-mismatch',
      retryable: false,
      status: 'unknown'
    })
  })

  it('rejects acceptance when the backend result omits a runtime identity', () => {
    expect(
      validateQuickEntryAcceptance(
        capturedRequest(),
        backendResult('streaming', { runtimeSessionId: '' }),
        AFTER_DISPATCH
      )
    ).toMatchObject({
      ok: false,
      reason: 'invalid-backend-result',
      retryable: false,
      status: 'unknown'
    })
  })

  it('rejects acceptance when the result correlation does not match', () => {
    expect(
      validateQuickEntryAcceptance(
        capturedRequest(),
        backendResult('streaming', { correlationId: 'qe-other' }),
        AFTER_DISPATCH
      )
    ).toMatchObject({
      ok: false,
      reason: 'correlation-mismatch',
      retryable: false,
      status: 'unknown'
    })
  })

  it('treats a proven pre-dispatch rejection as retryable', () => {
    expect(validateQuickEntryAcceptance(capturedRequest(), false, BEFORE_DISPATCH)).toEqual({
      ok: false,
      reason: 'no-prompt-dispatched',
      retryable: true,
      status: 'rejected'
    })
  })

  it('treats the same failure after dispatch as non-retryable unknown', () => {
    expect(validateQuickEntryAcceptance(capturedRequest(), false, AFTER_DISPATCH)).toEqual({
      ok: false,
      reason: 'invalid-backend-result',
      retryable: false,
      status: 'unknown'
    })
  })
})
