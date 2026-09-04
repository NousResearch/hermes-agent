/**
 * The renderer half of the `hermes:api` 404-envelope contract.
 *
 * `isMissingRestEndpoint` decides 404 from the error MESSAGE, so the envelope
 * that stops Electron logging control-flow 404s (electron/api-error-envelope.ts)
 * must reproduce that message byte-identically. The electron-project tests can't
 * import from `src/`, so the assertion against the SHIPPED predicate lives here.
 */
import { describe, expect, it } from 'vitest'

import { apiErrorEnvelope, apiErrorFromEnvelope } from '../../electron/api-error-envelope'

import { isMissingRestEndpoint } from './gateway-rpc'

function httpError(statusCode: number, body: string): Error & { statusCode: number } {
  const error = new Error(`${statusCode}: ${body}`) as Error & { statusCode: number }
  error.statusCode = statusCode

  return error
}

describe('isMissingRestEndpoint through the IPC error envelope', () => {
  it('still recognises a bare session 404 after the round trip', () => {
    const rethrown = apiErrorFromEnvelope(apiErrorEnvelope(httpError(404, '{"detail":"Session not found"}'))!)

    expect(isMissingRestEndpoint(rethrown)).toBe(true)
  })

  it('still recognises a missing route 404 after the round trip', () => {
    const rethrown = apiErrorFromEnvelope(
      apiErrorEnvelope(httpError(404, '{"detail":"No such API endpoint: /api/profiles/sessions/sidebar"}'))!
    )

    expect(isMissingRestEndpoint(rethrown)).toBe(true)
  })

  it('still recognises the wrapped remote-method shape', () => {
    const wrapped = httpError(404, '{"detail":"Session not found"}')
    wrapped.message = `Error invoking remote method 'hermes:api': Error: ${wrapped.message}`

    const rethrown = apiErrorFromEnvelope(apiErrorEnvelope(wrapped)!)

    expect(isMissingRestEndpoint(rethrown)).toBe(true)
  })

  it('does not turn a transient failure into a capability verdict', () => {
    // 500s and transport errors are never enveloped, so they keep rejecting and
    // must not read as "endpoint missing" either.
    expect(apiErrorEnvelope(httpError(500, 'boom'))).toBeNull()
    expect(isMissingRestEndpoint(httpError(500, 'boom'))).toBe(false)
    expect(isMissingRestEndpoint(new Error('socket hang up'))).toBe(false)
  })
})
