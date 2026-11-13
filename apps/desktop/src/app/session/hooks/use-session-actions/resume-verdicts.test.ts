import { describe, expect, it } from 'vitest'

import { isSessionGoneError } from './resume-verdicts'

describe('isSessionGoneError', () => {
  it('is true for gateway not_found refusals in structured and flattened shapes (#153047)', () => {
    // The boot-restore toast: JSON-RPC error frame flattened across the IPC
    // bridge (no prototype left) or kept as a JsonRpcGatewayError instance.
    expect(
      isSessionGoneError(Object.assign(new Error('not_found'), { code: 4001, data: { reason: 'not_found' } }))
    ).toBe(true)
    expect(isSessionGoneError({ code: 4007, message: 'session not found', data: { reason: 'not_found' } })).toBe(true)
    expect(isSessionGoneError(new Error("Error invoking remote method 'resume': Error: not_found"))).toBe(true)
    // tui_gateway/methods_session.py:815 answers a missing row with 4007 "session not found".
    expect(isSessionGoneError(Object.assign(new Error('session not found'), { code: 4007 }))).toBe(true)
  })

  it('is false when the wire code alone is not proof the session is gone (#153047)', () => {
    // 4001 also fronts invalid_params/permission-style refusals, and 4007 also
    // fronts the stale-binding refusal that tells the client to retry resume.
    // Neither may discard a session: only a not_found reason or gone wording does.
    expect(
      isSessionGoneError(Object.assign(new Error('invalid_params'), { code: 4001, data: { reason: 'invalid_params' } }))
    ).toBe(false)
    expect(isSessionGoneError(Object.assign(new Error('session no longer live; retry resume'), { code: 4007 }))).toBe(
      false
    )
  })
})