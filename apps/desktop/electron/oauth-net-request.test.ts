/**
 * Tests for OAuth-session Electron net.request helpers.
 *
 * Run with: node --test electron/oauth-net-request.test.ts
 */

import assert from 'node:assert/strict'

import { test } from 'vitest'

import { bindAbortSignal, serializeJsonBody, setJsonRequestHeaders } from './oauth-net-request'

test('serializeJsonBody returns undefined for absent bodies', () => {
  assert.equal(serializeJsonBody(undefined), undefined)
})

test('serializeJsonBody JSON-encodes request bodies', () => {
  const body = serializeJsonBody({ archived: true })
  assert.ok(Buffer.isBuffer(body))
  assert.equal(body.toString('utf8'), '{"archived":true}')
})

test('setJsonRequestHeaders does not set Electron-restricted Content-Length', () => {
  const headers = []

  const request = {
    setHeader(name, value) {
      headers.push([name, value])
    }
  }

  setJsonRequestHeaders(request)

  assert.deepEqual(headers, [['Content-Type', 'application/json']])
  assert.equal(
    headers.some(([name]) => name.toLowerCase() === 'content-length'),
    false
  )
})

test('bindAbortSignal aborts once and detaches cleanly', () => {
  const controller = new AbortController()
  let aborts = 0

  const detach = bindAbortSignal(controller.signal, () => {
    aborts += 1
  })

  controller.abort()
  controller.abort()
  assert.equal(aborts, 1)

  detach()
  assert.equal(aborts, 1)

  const alreadyAborted = new AbortController()
  alreadyAborted.abort()
  bindAbortSignal(alreadyAborted.signal, () => {
    aborts += 1
  })
  assert.equal(aborts, 2)
})
