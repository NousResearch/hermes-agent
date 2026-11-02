import assert from 'node:assert/strict'

import { test } from 'vitest'

import {
  buildFindClickExpression,
  CLICK_POLL_TIMEOUT_MS,
  evaluateJsonValue,
  pollUntil,
  POST_CLICK_READY_EXPRESSION
} from './click-session-helpers.mjs'

function cdpMessage(value) {
  // The shape send() resolves with: the whole CDP WebSocket message.
  return { id: 1, result: { result: { type: 'string', value } } }
}

test('buildFindClickExpression scrolls instantly and centered, then clicks', () => {
  const expr = buildFindClickExpression('Phaser particle')

  assert.ok(expr.includes("scrollIntoView({ behavior: 'auto', block: 'center' })"),
    'must use an instant, centered scroll (no default smooth scroll to drop)')
  assert.ok(expr.includes('found.click()'))
  assert.ok(expr.includes('JSON.stringify'))
  // The title must be embedded as JSON, so quotes/newlines in it cannot break
  // the expression.
  assert.ok(expr.includes(JSON.stringify('Phaser particle')))
})

test('evaluateJsonValue unwraps the nested CDP envelope', () => {
  assert.deepEqual(evaluateJsonValue(cdpMessage('{"found":true}')), { found: true })
})

test('evaluateJsonValue returns null on a truncated envelope, not undefined', () => {
  // The old script read `message.result.value` — one level short — and logged
  // undefined. Any missing level must yield null.
  assert.equal(evaluateJsonValue({}), null)
  assert.equal(evaluateJsonValue({ result: {} }), null)
  assert.equal(evaluateJsonValue({ result: { result: {} } }), null)
  assert.equal(evaluateJsonValue(cdpMessage('not-json')), null)
})

function fakeSend(answers) {
  let i = 0

  return async () => cdpMessage(answers[Math.min(i++, answers.length - 1)])
}

test('pollUntil resolves true as soon as the predicate answers true (no fixed sleep)', async () => {
  const send = fakeSend(['false', 'false', 'true'])

  const started = Date.now()
  assert.equal(await pollUntil(send, POST_CLICK_READY_EXPRESSION, { intervalMs: 5 }), true)
  // Three polls at 5ms apart: bounded by readiness, not a fixed 3000ms sleep.
  assert.ok(Date.now() - started < 500)
})

test('pollUntil returns false at the bound instead of hanging when never ready', async () => {
  const send = fakeSend(['false'])

  const started = Date.now()
  assert.equal(
    await pollUntil(send, POST_CLICK_READY_EXPRESSION, { timeoutMs: 200, intervalMs: 50 }),
    false
  )
  assert.ok(Date.now() - started >= 150, 'waited for the full bound')
  assert.ok(Date.now() - started < 3000, 'did not fall back to a fixed sleep')
})

test('default poll bound matches the old sleep ceiling', () => {
  // The readiness poll replaces the fixed 3000ms sleep: same ceiling, but it
  // exits early when the session is ready.
  assert.equal(CLICK_POLL_TIMEOUT_MS, 3000)
})
