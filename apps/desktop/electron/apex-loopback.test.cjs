/**
 * Tests for electron/apex-loopback.cjs — the minimal loopback HTTP server behind
 * the ApexNodes Desktop browser-login flows ("用 Google 登录" / "用 APEX 登录").
 *
 * Run with: node --test electron/apex-loopback.test.cjs
 * (Wired into npm test:desktop:platforms in package.json.)
 *
 * apex-loopback.cjs is electron-free (node:http + node:crypto), so we exercise
 * the real server over a real socket — bind it, hit 127.0.0.1:<port>/cb with the
 * Node http client, and assert the `result` promise settles correctly.
 */

const test = require('node:test')
const assert = require('node:assert/strict')
const http = require('node:http')

const { generateState, startLoopbackLogin, DEFAULT_TIMEOUT_MS } = require('./apex-loopback.cjs')

// Fire a GET at the loopback callback and resolve with { statusCode, body }.
// agent:false → no keep-alive pooling, so the client socket closes right after
// the response (mirrors how a real browser navigation behaves and lets the
// server's handle release cleanly between tests).
function hitLoopback(port, pathAndQuery) {
  return new Promise((resolve, reject) => {
    const req = http.request(
      { host: '127.0.0.1', port, path: pathAndQuery, method: 'GET', agent: false },
      res => {
        const chunks = []
        res.on('data', c => chunks.push(c))
        res.on('end', () => resolve({ statusCode: res.statusCode, body: Buffer.concat(chunks).toString('utf8') }))
      }
    )
    req.on('error', reject)
    req.end()
  })
}

test('generateState returns a long, URL-safe, unique token', () => {
  const a = generateState()
  const b = generateState()
  assert.notEqual(a, b)
  assert.ok(a.length >= 24)
  assert.match(a, /^[A-Za-z0-9_-]+$/)
})

test('startLoopbackLogin binds 127.0.0.1 with a /cb redirect_uri and a state', async () => {
  const lb = await startLoopbackLogin()
  try {
    assert.ok(lb.port > 0)
    assert.equal(lb.redirectUri, `http://127.0.0.1:${lb.port}/cb`)
    assert.ok(lb.state.length >= 24)
  } finally {
    lb.close()
    await lb.result.catch(() => {})
  }
})

test('resolves with the token on a valid /cb callback (matching state)', async () => {
  const lb = await startLoopbackLogin()
  const res = await hitLoopback(lb.port, `/cb?token=jwt.success&state=${encodeURIComponent(lb.state)}`)
  assert.equal(res.statusCode, 200)
  assert.match(res.body, /登录成功/)
  const outcome = await lb.result
  assert.deepEqual(outcome, { token: 'jwt.success' })
})

test('rejects (state_mismatch) on a CSRF mismatch and serves a 400 page', async () => {
  const lb = await startLoopbackLogin()
  // Attach the rejection expectation BEFORE triggering the callback: the flow
  // rejects synchronously while servicing the request, so a handler must already
  // be on lb.result or Node flags it as an unhandled rejection.
  const rejected = assert.rejects(lb.result, err => err.reason === 'state_mismatch')
  const res = await hitLoopback(lb.port, '/cb?token=jwt.x&state=evil')
  assert.equal(res.statusCode, 400)
  assert.match(res.body, /登录失败/)
  await rejected
})

test('rejects (missing_token) when state matches but no token is present', async () => {
  const lb = await startLoopbackLogin()
  const rejected = assert.rejects(lb.result, err => err.reason === 'missing_token')
  const res = await hitLoopback(lb.port, `/cb?state=${encodeURIComponent(lb.state)}`)
  assert.equal(res.statusCode, 400)
  await rejected
})

test('ignores a non-/cb request (favicon) without settling, then resolves on /cb', async () => {
  const lb = await startLoopbackLogin()
  // The browser auto-requests /favicon.ico; it must not fail the flow.
  const favicon = await hitLoopback(lb.port, '/favicon.ico')
  assert.equal(favicon.statusCode, 204)

  // The real callback still resolves afterward.
  await hitLoopback(lb.port, `/cb?token=jwt.after.favicon&state=${encodeURIComponent(lb.state)}`)
  const outcome = await lb.result
  assert.equal(outcome.token, 'jwt.after.favicon')
})

test('close() rejects an in-flight flow with reason "aborted"', async () => {
  const lb = await startLoopbackLogin()
  const rejected = assert.rejects(lb.result, err => err.reason === 'aborted')
  lb.close()
  await rejected
})

test('the watchdog rejects with "timeout" when the browser never returns', async () => {
  const lb = await startLoopbackLogin({ timeoutMs: 30 })
  await assert.rejects(lb.result, err => err.reason === 'timeout')
})

test('DEFAULT_TIMEOUT_MS is a generous, browser-friendly window', () => {
  assert.ok(DEFAULT_TIMEOUT_MS >= 60_000)
})
