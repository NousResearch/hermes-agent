import { EventEmitter } from 'node:events'

import { describe, expect, it } from 'vitest'

import { type OauthResponseLike, wireOauthSessionResponse } from './oauth-session-response'

/**
 * Behavior regression for #72530: body stream errors emitted AFTER the
 * `response` event (e.g. net::ERR_CONTENT_LENGTH_MISMATCH when the body is
 * truncated after headers) must reject the promise. These errors arrive on the
 * IncomingMessage, not the ClientRequest — fetchJson/fetchPublicJson already
 * handle this and fetchJsonViaOauthSession must too.
 *
 * We drive the exact response-handling unit main.ts wires into
 * request.on('response', ...) with a fake IncomingMessage, so no real socket
 * or electron module is needed and we assert observable behavior (resolve vs
 * reject) rather than source text.
 */
function makeFakeResponse(statusCode: number, headers: Record<string, string> = {}): EventEmitter & OauthResponseLike {
  const res = new EventEmitter() as EventEmitter & OauthResponseLike

  ;(res as { statusCode?: number }).statusCode = statusCode
  ;(res as { headers: Record<string, string> }).headers = headers

  return res
}

function wire(res: EventEmitter & OauthResponseLike) {
  let resolved: unknown
  let rejected: Error | undefined
  let cleared = 0

  wireOauthSessionResponse(res, {
    url: 'https://gw.example.com/api',
    isTimedOut: () => false,
    clearTimer: () => {
      cleared += 1
    },
    resolve: value => {
      resolved = value
    },
    reject: error => {
      rejected = error
    },
  })

  return {
    get resolved() {
      return resolved
    },
    get rejected() {
      return rejected
    },
    get cleared() {
      return cleared
    },
  }
}

describe('wireOauthSessionResponse body error handling (#72530)', () => {
  it('rejects when the response emits an error after data (truncated body)', () => {
    const res = makeFakeResponse(200, { 'content-type': 'application/json' })
    const state = wire(res)

    res.emit('data', Buffer.from('{"partial":'))
    const err = new Error('net::ERR_CONTENT_LENGTH_MISMATCH')
    res.emit('error', err)

    expect(state.rejected).toBe(err)
    expect(state.resolved).toBeUndefined()
    expect(state.cleared).toBe(1)
  })

  it('resolves parsed JSON on a clean end', () => {
    const res = makeFakeResponse(200, { 'content-type': 'application/json' })
    const state = wire(res)

    res.emit('data', Buffer.from('{"ok":true}'))
    res.emit('end')

    expect(state.rejected).toBeUndefined()
    expect(state.resolved).toEqual({ ok: true })
  })

  it('rejects with statusCode on HTTP errors', () => {
    const res = makeFakeResponse(503, {})
    const state = wire(res)

    res.emit('data', Buffer.from('unavailable'))
    res.emit('end')

    expect(state.resolved).toBeUndefined()
    expect((state.rejected as Error & { statusCode?: number }).statusCode).toBe(503)
  })

  it('rejects HTML bodies as non-JSON', () => {
    const res = makeFakeResponse(200, { 'content-type': 'text/html' })
    const state = wire(res)

    res.emit('data', Buffer.from('<!doctype html><html></html>'))
    res.emit('end')

    expect(state.resolved).toBeUndefined()
    expect(state.rejected?.message).toMatch(/got HTML/)
  })
})
