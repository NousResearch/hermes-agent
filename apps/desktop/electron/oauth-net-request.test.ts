/**
 * Tests for OAuth-session Electron net.request helpers.
 *
 * Run with: node --test electron/oauth-net-request.test.ts
 */

import assert from 'node:assert/strict'
import { EventEmitter } from 'node:events'

import { afterEach, describe, expect, test, vi } from 'vitest'

import { fetchJsonViaOauthSession, serializeJsonBody, setRequestContentType } from './oauth-net-request'

test('serializeJsonBody returns undefined for absent bodies', () => {
  assert.equal(serializeJsonBody(undefined), undefined)
})

test('serializeJsonBody JSON-encodes request bodies', () => {
  const body = serializeJsonBody({ archived: true })
  assert.ok(Buffer.isBuffer(body))
  assert.equal(body.toString('utf8'), '{"archived":true}')
})

test('setRequestContentType does not set Electron-restricted Content-Length', () => {
  const headers = []

  const request = {
    setHeader(name, value) {
      headers.push([name, value])
    }
  }

  setRequestContentType(request)

  assert.deepEqual(headers, [['Content-Type', 'application/json']])
  assert.equal(
    headers.some(([name]) => name.toLowerCase() === 'content-length'),
    false
  )
})

test('setRequestContentType accepts multipart content types', () => {
  const headers = []

  const request = {
    setHeader(name, value) {
      headers.push([name, value])
    }
  }

  setRequestContentType(request, 'multipart/form-data; boundary=abc')

  assert.deepEqual(headers, [['Content-Type', 'multipart/form-data; boundary=abc']])
})

class FakeRequest extends EventEmitter {
  aborted = false
  ended = false
  errorOnAbort: Error | null = null
  headers: Array<[string, string]> = []
  writes: Buffer[] = []

  abort() {
    this.aborted = true

    if (this.errorOnAbort) {
      this.emit('error', this.errorOnAbort)
    }
  }

  end() {
    this.ended = true
  }

  setHeader(name, value) {
    this.headers.push([name, value])
  }

  write(body) {
    this.writes.push(Buffer.from(body))
  }
}

function response(statusCode = 200, headers = {}) {
  return Object.assign(new EventEmitter(), { headers, statusCode })
}

function deps(overrides: any = {}) {
  const requests: FakeRequest[] = []
  const requestOptions: any[] = []

  const net = {
    request(options) {
      requestOptions.push(options)
      const request = new FakeRequest()
      requests.push(request)

      return request
    }
  }

  return {
    defaultTimeoutMs: 15_000,
    multipartBody(upload) {
      return {
        body: Buffer.from(upload.bytes),
        contentType: `multipart/form-data; boundary=${upload.boundary || 'boundary'}`
      }
    },
    net,
    requestOptions,
    requests,
    resolveTimeoutMs(value, fallback) {
      return Number(value) > 0 ? Number(value) : fallback
    },
    session: { partition: 'persist:hermes-remote-oauth' },
    ...overrides
  }
}

afterEach(() => {
  vi.useRealTimers()
})

describe('fetchJsonViaOauthSession', () => {
  test('sends GET requests through the OAuth session without body or entity headers', async () => {
    const d = deps()
    const promise = fetchJsonViaOauthSession('http://100.118.5.105:9120/api/plugins/kanban/board', {}, d)

    expect(d.requestOptions).toEqual([
      {
        method: 'GET',
        redirect: 'follow',
        session: d.session,
        url: 'http://100.118.5.105:9120/api/plugins/kanban/board',
        useSessionCookies: true
      }
    ])
    expect(d.requests[0].headers).toEqual([])
    expect(d.requests[0].writes).toEqual([])
    expect(d.requests[0].ended).toBe(true)

    const res = response(200, { 'content-type': 'application/json' })
    d.requests[0].emit('response', res)
    res.emit('data', Buffer.from('{"columns":[]}'))
    res.emit('end')

    await expect(promise).resolves.toEqual({ columns: [] })
  })

  test('sends JSON bodies with Content-Type and no restricted Content-Length', async () => {
    const d = deps()

    const promise = fetchJsonViaOauthSession(
      'https://example.test/api/plugins/kanban/tasks',
      { body: { title: 'x' }, method: 'POST' },
      d
    )

    expect(d.requestOptions[0]).toMatchObject({
      method: 'POST',
      session: d.session,
      useSessionCookies: true
    })
    expect(d.requests[0].headers).toEqual([['Content-Type', 'application/json']])
    expect(d.requests[0].headers.some(([name]) => name.toLowerCase() === 'content-length')).toBe(false)
    expect(d.requests[0].writes.map(body => body.toString('utf8'))).toEqual(['{"title":"x"}'])

    const res = response(200, { 'Content-Type': 'application/json' })
    d.requests[0].emit('response', res)
    res.emit('data', Buffer.from('{"ok":true}'))
    res.emit('end')

    await expect(promise).resolves.toEqual({ ok: true })
  })

  test('preserves multipart upload bodies and content type', async () => {
    const d = deps()
    const bytes = new Uint8Array([1, 2, 3]).buffer

    const promise = fetchJsonViaOauthSession(
      'http://127.0.0.1:9120/api/plugins/kanban/tasks/t_1/attachments',
      {
        method: 'POST',
        upload: { boundary: 'upload-boundary', bytes, filename: 'proof.txt' }
      },
      d
    )

    expect(d.requests[0].headers).toEqual([['Content-Type', 'multipart/form-data; boundary=upload-boundary']])
    expect([...d.requests[0].writes[0]]).toEqual([1, 2, 3])

    const res = response(200, { 'content-type': 'application/json' })
    d.requests[0].emit('response', res)
    res.emit('data', Buffer.from('{"attachment":{"id":"a_1"}}'))
    res.emit('end')

    await expect(promise).resolves.toEqual({ attachment: { id: 'a_1' } })
  })

  test('clears timeout after request errors', async () => {
    vi.useFakeTimers()
    const d = deps()
    const promise = fetchJsonViaOauthSession('http://127.0.0.1:9120/api/status', { timeoutMs: 25 }, d)

    d.requests[0].emit('error', new Error('socket closed'))
    await expect(promise).rejects.toThrow('socket closed')

    await vi.advanceTimersByTimeAsync(25)
    expect(d.requests[0].aborted).toBe(false)
  })

  test('aborts and ignores late errors after timeout', async () => {
    vi.useFakeTimers()
    const d = deps()
    const promise = fetchJsonViaOauthSession('http://127.0.0.1:9120/api/status', { timeoutMs: 25 }, d)
    const rejection = expect(promise).rejects.toThrow('Timed out connecting to Hermes backend after 25ms')

    await vi.advanceTimersByTimeAsync(25)
    await rejection
    expect(d.requests[0].aborted).toBe(true)

    d.requests[0].emit('error', new Error('late socket closed'))
  })

  test('preserves the timeout error when abort emits an error synchronously', async () => {
    vi.useFakeTimers()
    const d = deps()
    const promise = fetchJsonViaOauthSession('http://127.0.0.1:9120/api/status', { timeoutMs: 25 }, d)

    d.requests[0].errorOnAbort = new Error('abort emitted synchronously')
    const rejection = expect(promise).rejects.toThrow('Timed out connecting to Hermes backend after 25ms')

    await vi.advanceTimersByTimeAsync(25)
    await rejection
    expect(d.requests[0].aborted).toBe(true)
  })

  test('clears timeout after HTTP error responses', async () => {
    vi.useFakeTimers()
    const d = deps()
    const promise = fetchJsonViaOauthSession('http://127.0.0.1:9120/api/plugins/kanban/board', { timeoutMs: 25 }, d)
    const res = response(401, { 'content-type': 'application/json' })

    d.requests[0].emit('response', res)
    res.emit('data', Buffer.from('{"detail":"unauthorized"}'))
    res.emit('end')

    await expect(promise).rejects.toMatchObject({ statusCode: 401 })
    await vi.advanceTimersByTimeAsync(25)
    expect(d.requests[0].aborted).toBe(false)
  })
})
