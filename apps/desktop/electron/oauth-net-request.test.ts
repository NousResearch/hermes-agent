/**
 * Tests for OAuth-session Electron net.request helpers.
 *
 * Run with: node --test electron/oauth-net-request.test.ts
 */

import assert from 'node:assert/strict'

import { test } from 'vitest'

import { serializeJsonBody, setJsonRequestHeaders } from './oauth-net-request'

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

test('setJsonRequestHeaders derives Origin for a state-changing POST', () => {
  const headers = []
  const request = { setHeader: (name, value) => headers.push([name, value]) }

  setJsonRequestHeaders(request, {
    method: 'POST',
    url: 'https://gateway.example:3738/hermes/api/auth/ws-ticket'
  })

  assert.deepEqual(headers, [
    ['Content-Type', 'application/json'],
    ['Origin', 'https://gateway.example:3738']
  ])
})

test('setJsonRequestHeaders omits Origin for a safe GET', () => {
  const headers = []
  const request = { setHeader: (name, value) => headers.push([name, value]) }

  setJsonRequestHeaders(request, {
    method: 'GET',
    url: 'https://gateway.example:3738/hermes/api/status'
  })

  assert.deepEqual(headers, [['Content-Type', 'application/json']])
})
