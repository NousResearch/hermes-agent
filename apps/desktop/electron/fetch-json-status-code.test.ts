/**
 * Regression coverage for #fetchJson / #fetchPublicJson error shape.
 *
 * isGatewayAuthRejection() classifies an auth failure SOLELY from
 * err.statusCode, and gatewayTicketFailure()'s own comment already states
 * "the fetch layer attaches err.statusCode". These two fetchers only embedded
 * the status in the message string, so a gateway 401/403 reached the classifier
 * with statusCode undefined, was treated as a transport failure, and surfaced
 * to the user as "Could not reach the remote Hermes gateway while refreshing
 * its WebSocket ticket" — while the gateway was healthy and the real fault was
 * an expired session.
 *
 * These fetchers close over main-process singletons (http/https, the session
 * token header), so their shape is asserted on source — the same approach as
 * gateway-file-download-transport.test.ts — while the consequence for the
 * classifier is covered behaviorally below.
 */

import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

import { test } from 'vitest'

import { gatewayTicketFailure, isGatewayAuthRejection } from './connection-config'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const source = fs.readFileSync(path.join(__dirname, 'main.ts'), 'utf8').replace(/\r\n/g, '\n')

function extract(startMarker: string, endMarker: string): string {
  const start = source.indexOf(startMarker)
  assert.notEqual(start, -1, `${startMarker} should exist`)
  const end = source.indexOf(endMarker, start + startMarker.length)
  assert.notEqual(end, -1, `boundary after ${startMarker} should exist`)

  return source.slice(start, end)
}

test('fetchJson attaches the HTTP status as err.statusCode, not only in the message', () => {
  const fn = extract('function fetchJson(url, token, options', '\nfunction fetchPublicJson')

  assert.match(fn, /err\.statusCode = res\.statusCode \|\| 500/)
})

test('fetchPublicJson attaches the HTTP status as err.statusCode', () => {
  const fn = extract('function fetchPublicJson(url, options', '\nfunction ')

  assert.match(fn, /err\.statusCode = res\.statusCode \|\| 500/)
})

test('a 401 carrying statusCode routes to reauth, not to a transport failure', () => {
  // The shape the patched fetchers reject with.
  const err = new Error('401: {"error":"session_expired"}') as any
  err.statusCode = 401

  assert.equal(isGatewayAuthRejection(err), true)

  const wrapped = gatewayTicketFailure(err, 'Session expired — sign in again.', 'Could not reach the gateway.') as any

  assert.equal(wrapped.needsOauthLogin, true)
  assert.equal(wrapped.message, 'Session expired — sign in again.')
  assert.equal(wrapped.statusCode, 401)
})

test('the pre-fix shape (status only in the message) is what misreported a 401', () => {
  // Documents the regression this guards: no statusCode property means the
  // classifier cannot see the 401 and the user is told the gateway is down.
  const legacy = new Error('401: {"error":"session_expired"}') as any

  assert.equal(isGatewayAuthRejection(legacy), false)

  const wrapped = gatewayTicketFailure(
    legacy,
    'Session expired — sign in again.',
    'Could not reach the gateway.'
  ) as any

  assert.equal(wrapped.needsOauthLogin, undefined)
  assert.equal(wrapped.message, 'Could not reach the gateway.')
})
