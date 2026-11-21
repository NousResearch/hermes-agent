import { describe, expect, it } from 'vitest'

import { isNoRuntimeToStopError, isSessionGoneError } from './resume-verdicts'

const error = (message: string) => new Error(message)

describe('isNoRuntimeToStopError', () => {
  it('tolerates a reaped runtime the gateway reports as not_found', () => {
    // The delete path interrupts the runtime it still remembers first; an
    // idle-reaped runtime answers the bare JSON-RPC verdict `not_found`.
    expect(isNoRuntimeToStopError(error('not_found'))).toBe(true)
  })

  it('tolerates the detached-runtime warning the gateway words as not in memory', () => {
    expect(
      isNoRuntimeToStopError(
        error('not in memory (detached/reaped runtime; client should resume the stored session)')
      )
    ).toBe(true)
  })

  it('tolerates a superseded mount, which can never run a turn', () => {
    expect(isNoRuntimeToStopError(error('JsonRpcGatewayError: session_superseded'))).toBe(true)
  })

  it('keeps the REST 404 and stored-id verdicts', () => {
    expect(isNoRuntimeToStopError(error('404: {"detail":"not_found"}'))).toBe(true)
    expect(isNoRuntimeToStopError(error('Session not found'))).toBe(true)
  })

  it('still refuses to continue on a transient or a real refusal', () => {
    expect(isNoRuntimeToStopError(error('connect ECONNREFUSED 127.0.0.1:8642'))).toBe(false)
    expect(isNoRuntimeToStopError(error('Timed out connecting to Hermes backend after 8000ms'))).toBe(false)
    expect(isNoRuntimeToStopError(error('session_busy'))).toBe(false)
    expect(isNoRuntimeToStopError(error('permission_denied'))).toBe(false)
    expect(isNoRuntimeToStopError(undefined)).toBe(false)
  })
})

describe('isSessionGoneError', () => {
  it('stays narrowed to the owning backend 404 verdicts', () => {
    // Resume uses this to decide a session may be DISCARDED, so a bare
    // not_found must not qualify: a cross-profile open races the gateway swap
    // and can produce it for a perfectly alive session (#88540).
    expect(isSessionGoneError(error('not_found'))).toBe(false)
    expect(isSessionGoneError(error('404: {"detail":"not_found"}'))).toBe(true)
    expect(isSessionGoneError(error('Session not found'))).toBe(true)
  })
})
