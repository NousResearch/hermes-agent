import { Buffer } from 'node:buffer'
import { EventEmitter } from 'node:events'

import { describe, expect, it, vi } from 'vitest'

import { httpStatusError, readStatusCode } from './api-transport'
import { wireOauthSessionResponse } from './oauth-session-response'

function makeResponse(statusCode = 200, headers: Record<string, string | undefined> = {}) {
  return Object.assign(new EventEmitter(), { statusCode, headers })
}

function wire(res: ReturnType<typeof makeResponse>) {
  const resolve = vi.fn()
  const reject = vi.fn()
  const clearTimer = vi.fn()
  let timedOut = false

  wireOauthSessionResponse(res, {
    url: 'https://gw.example.com/api',
    isTimedOut: () => timedOut,
    clearTimer,
    resolve,
    reject,
  })

  return {
    resolve,
    reject,
    clearTimer,
    timeOut: () => {
      timedOut = true
    },
  }
}

describe('OAuth session response', () => {
  it('settles once when a response body fails after headers', () => {
    const res = makeResponse()
    const state = wire(res)
    const error = new Error('net::ERR_CONTENT_LENGTH_MISMATCH')

    res.emit('data', Buffer.from('{"partial":'))
    expect(() => res.emit('error', error)).not.toThrow()

    // Neither another loader failure nor end may settle a failed body again.
    expect(() => res.emit('error', new Error('late failure'))).not.toThrow()
    res.emit('end')

    expect(state.reject).toHaveBeenCalledExactlyOnceWith(error)
    expect(state.resolve).not.toHaveBeenCalled()
    expect(state.clearTimer).toHaveBeenCalledTimes(1)
    expect(() => res.emit('data', null)).not.toThrow()

    const timedOutRes = makeResponse()
    const timedOutState = wire(timedOutRes)
    timedOutRes.emit('data', Buffer.from('{"partial":'))
    timedOutState.timeOut()
    expect(() => timedOutRes.emit('error', error)).not.toThrow()
    timedOutRes.emit('end')
    expect(() => timedOutRes.emit('data', null)).not.toThrow()
    expect(timedOutState.reject).not.toHaveBeenCalled()
    expect(timedOutState.resolve).not.toHaveBeenCalled()
    expect(timedOutState.clearTimer).not.toHaveBeenCalled()
  })

  it('preserves JSON and HTTP error contracts when end wins settlement', () => {
    const cases = [
      { status: 200, body: '{"ok":"✓"}', headers: {}, value: { ok: '✓' } },
      { status: 204, body: '', headers: {}, value: null },
      { status: 401, body: '{"error":"session_expired"}', headers: {} },
      { status: 403, body: '', headers: {} },
      { status: 503, body: 'unavailable', headers: {} },
      { status: 0, body: 'missing status', headers: {} },
      { status: 200, body: ' \n<!doctype html><html></html>', headers: {}, error: /got HTML/ },
      { status: 200, body: '{}', headers: { 'Content-Type': 'text/html' }, error: /got HTML/ },
      { status: 200, body: '{"partial":', headers: {}, error: /Invalid JSON/ },
    ]

    for (const entry of cases) {
      const res = makeResponse(entry.status, entry.headers)
      const state = wire(res)
      // Splitting every byte also covers UTF-8 characters split across chunks.
      for (const byte of Buffer.from(entry.body)) {
        res.emit('data', Buffer.from([byte]))
      }
      res.emit('end')

      // Late terminal events must not overwrite either success or rejection.
      expect(() => res.emit('error', new Error('late loader failure'))).not.toThrow()
      res.emit('end')
      expect(() => res.emit('data', null)).not.toThrow()
      expect(state.clearTimer).toHaveBeenCalledTimes(1)

      if ('value' in entry) {
        expect(state.resolve).toHaveBeenCalledExactlyOnceWith(entry.value)
        expect(state.reject).not.toHaveBeenCalled()
      } else {
        expect(state.resolve).not.toHaveBeenCalled()
        expect(state.reject).toHaveBeenCalledTimes(1)
        const error = state.reject.mock.calls[0][0]
        if (entry.error) {
          expect(error.message).toMatch(entry.error)
          expect(readStatusCode(error)).toBeNaN()
        } else {
          const expected = httpStatusError(entry.status, entry.body)
          expect(error.message).toBe(expected.message)
          expect(readStatusCode(error)).toBe(readStatusCode(expected))
        }
      }
    }
  })
})
