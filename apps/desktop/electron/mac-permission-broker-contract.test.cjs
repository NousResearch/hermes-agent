'use strict'

const assert = require('node:assert/strict')
const test = require('node:test')

const {
  MAC_PERMISSION_BROKER_METHODS,
  createBrokerRequest,
  redactForBrokerAudit,
  stableStringify,
  verifyBrokerRequest
} = require('./mac-permission-broker-contract.cjs')

const TOKEN = '0123456789abcdef0123456789abcdef'

test('method registry only exposes explicit broker capabilities', () => {
  assert.deepEqual(Object.keys(MAC_PERMISSION_BROKER_METHODS).sort(), [
    'automation.appleEvent',
    'mic.status',
    'notification.send',
    'permission.openSettings',
    'permission.status',
    'screen.record',
    'screen.snapshot',
    'system.run.approved',
    'ui.click',
    'ui.type'
  ])
  assert.equal(MAC_PERMISSION_BROKER_METHODS['ui.click'].tcc, 'Accessibility')
  assert.equal(MAC_PERMISSION_BROKER_METHODS['screen.snapshot'].tcc, 'ScreenCapture')
})

test('stableStringify canonicalizes object key order for cross-language signing', () => {
  const left = { z: 1, a: { y: true, b: [3, { d: 'x', c: 'y' }] } }
  const right = { a: { b: [3, { c: 'y', d: 'x' }], y: true }, z: 1 }
  assert.equal(stableStringify(left), stableStringify(right))
})

test('createBrokerRequest signs and verifyBrokerRequest accepts a fresh request', () => {
  const request = createBrokerRequest({
    method: 'screen.snapshot',
    params: { displayId: 1 },
    token: TOKEN,
    now: 1_000,
    ttlMs: 10_000,
    nonce: 'nonce-1',
    id: 'request-1'
  })

  const result = verifyBrokerRequest(request, TOKEN, { now: 2_000 })

  assert.equal(result.ok, true)
  assert.equal(result.id, 'request-1')
  assert.equal(result.method, 'screen.snapshot')
  assert.deepEqual(result.params, { displayId: 1 })
  assert.equal(result.capability.tcc, 'ScreenCapture')
})

test('verifyBrokerRequest rejects method tampering after signing', () => {
  const request = createBrokerRequest({ method: 'permission.status', token: TOKEN, now: 1_000, nonce: 'nonce-2' })
  request.method = 'ui.click'

  const result = verifyBrokerRequest(request, TOKEN, { now: 2_000 })

  assert.equal(result.ok, false)
  assert.match(result.error, /signature mismatch/)
})

test('verifyBrokerRequest rejects unsupported methods before execution', () => {
  const request = createBrokerRequest({ method: 'permission.status', token: TOKEN, now: 1_000, nonce: 'nonce-3' })
  request.method = 'system.run.raw'

  const result = verifyBrokerRequest(request, TOKEN, { now: 2_000 })

  assert.equal(result.ok, false)
  assert.match(result.error, /unsupported broker method/)
})

test('verifyBrokerRequest rejects expired requests', () => {
  const request = createBrokerRequest({ method: 'ui.type', token: TOKEN, now: 1_000, ttlMs: 500, nonce: 'nonce-4' })

  const result = verifyBrokerRequest(request, TOKEN, { now: 10_000, clockSkewMs: 0 })

  assert.equal(result.ok, false)
  assert.match(result.error, /expired/)
})

test('verifyBrokerRequest rejects replayed nonces when caller provides a nonce cache', () => {
  const seenNonces = new Set()
  const request = createBrokerRequest({ method: 'notification.send', token: TOKEN, now: 1_000, nonce: 'nonce-5' })

  assert.equal(verifyBrokerRequest(request, TOKEN, { now: 2_000, seenNonces }).ok, true)
  const replay = verifyBrokerRequest(request, TOKEN, { now: 2_500, seenNonces })

  assert.equal(replay.ok, false)
  assert.match(replay.error, /replayed/)
})

test('redactForBrokerAudit removes nested sensitive values without losing structure', () => {
  const redacted = redactForBrokerAudit({
    method: 'automation.appleEvent',
    params: {
      targetBundleId: 'com.apple.finder',
      token: 'secret-token',
      nested: { password: 'pw', ok: true }
    }
  })

  assert.deepEqual(redacted, {
    method: 'automation.appleEvent',
    params: {
      targetBundleId: 'com.apple.finder',
      token: '[REDACTED]',
      nested: { password: '[REDACTED]', ok: true }
    }
  })
})

test('createBrokerRequest refuses short tokens and unknown methods', () => {
  assert.throws(() => createBrokerRequest({ method: 'permission.status', token: 'short' }), /at least 32/)
  assert.throws(() => createBrokerRequest({ method: 'raw.exec', token: TOKEN }), /unsupported macOS broker method/)
})
