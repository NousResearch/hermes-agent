/**
 * The 404-envelope contract for the `hermes:api` IPC channel.
 *
 * Guards the two renderer-visible invariants a "stop logging this" fix could
 * silently break: the message must survive byte-identically (404 detection is
 * message-based), and only 404 may be enveloped — everything else has to keep
 * rejecting so real faults stay logged.
 */
import { describe, expect, it } from 'vitest'

import {
  API_ERROR_ENVELOPE,
  apiErrorEnvelope,
  apiErrorFromEnvelope,
  isApiErrorEnvelope
} from './api-error-envelope'

// The renderer's real predicate lives in src/, outside this tsconfig project.
// The end-to-end assertion against the shipped `isMissingRestEndpoint` lives in
// the ui project (src/lib/gateway-rpc.envelope.test.ts); here we only pin the
// message pattern it keys on.
const READS_AS_404 = /(?:^\s*|error:\s*)404\b/i

function httpError(statusCode: number, body: string): Error & { statusCode: number } {
  const error = new Error(`${statusCode}: ${body}`) as Error & { statusCode: number }
  error.statusCode = statusCode

  return error
}

const SESSION_404 = () => httpError(404, '{"detail":"Session not found"}')

describe('apiErrorEnvelope', () => {
  it('envelopes a 404 so the handler can resolve instead of rejecting', () => {
    const envelope = apiErrorEnvelope(SESSION_404())

    expect(envelope).not.toBeNull()
    expect(envelope?.statusCode).toBe(404)
    expect(isApiErrorEnvelope(envelope)).toBe(true)
  })

  it('leaves every other failure rejecting, so real faults stay logged', () => {
    for (const error of [
      httpError(500, 'boom'),
      httpError(401, 'nope'),
      httpError(502, 'bad gateway'),
      new Error('socket hang up'),
      Object.assign(new Error('refused'), { code: 'ECONNREFUSED' })
    ]) {
      expect(apiErrorEnvelope(error)).toBeNull()
    }
  })

  it('does not treat an ordinary REST payload as an envelope', () => {
    expect(isApiErrorEnvelope({ id: 'x', title: 'a session' })).toBe(false)
    expect(isApiErrorEnvelope(null)).toBe(false)
    expect(isApiErrorEnvelope('404')).toBe(false)
    expect(isApiErrorEnvelope({ [API_ERROR_ENVELOPE]: true })).toBe(false)
  })
})

describe('apiErrorFromEnvelope', () => {
  it('reproduces the rejection message byte-identically', () => {
    const original = SESSION_404()
    const envelope = apiErrorEnvelope(original)
    const rethrown = apiErrorFromEnvelope(envelope!)

    expect(rethrown).toBeInstanceOf(Error)
    expect(rethrown.message).toBe(original.message)
    expect((rethrown as Error & { statusCode?: number }).statusCode).toBe(404)
  })

  it('keeps message-based 404 detection working through the envelope', () => {
    // isMissingRestEndpoint is the real consumer: a changed message shape would
    // silently turn "route/row missing" into "unknown failure" at every caller.
    // Asserted against the shipped predicate in the ui-project companion test.
    const rethrown = apiErrorFromEnvelope(apiErrorEnvelope(SESSION_404())!)

    expect(READS_AS_404.test(rethrown.message)).toBe(true)
  })

  it('survives the wrapped remote-method shape the renderer also parses', () => {
    const wrapped = httpError(404, '{"detail":"No such API endpoint: /api/x"}')
    wrapped.message = `Error invoking remote method 'hermes:api': Error: ${wrapped.message}`

    const rethrown = apiErrorFromEnvelope(apiErrorEnvelope(wrapped)!)

    expect(rethrown.message).toBe(wrapped.message)
    expect(READS_AS_404.test(rethrown.message)).toBe(true)
  })
})
