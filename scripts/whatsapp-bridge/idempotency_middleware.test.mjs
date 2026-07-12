/**
 * Integration tests for the Idempotency-Key middleware in the WhatsApp bridge.
 *
 * These tests simulate the Express middleware that wraps the mutating
 * routes (/send, /send-media, /send-poll, /send-location, /edit) with
 * idempotency deduplication. They do NOT require a live Baileys socket —
 * the "send" function is mocked.
 *
 * Key scenarios:
 *   1. No Idempotency-Key header → passes through (no-op)
 *   2. Same key+route+payload → cached response returned without re-send
 *   3. Same key, different payload → 409 conflict
 *   4. Concurrent identical requests → single send
 *   5. Different routes with same key each execute independently
 *   6. GET routes are NOT affected (non-mutating)
 *   7. Non-200 status and body are cached exactly
 *   8. No PII in any observable output
 */

import { strict as assert } from 'node:assert';
import { createIdempotencyStore } from './idempotency_store.js';
import { createIdempotencyMiddleware } from './idempotency_middleware.js';

// ------------------------------------------------------------------
// Helper: mock Express req/res
// ------------------------------------------------------------------
function mockReq(method, route, body = {}, headers = {}) {
  return {
    method,
    path: route,
    body,
    headers: { ...headers },
  };
}

function mockRes() {
  const res = {
    _status: null,
    _body: null,
    statusCode: 200,
    status(code) { this._status = code; this.statusCode = code; return this; },
    json(body) { this._body = body; return this; },
    set(key, val) { return this; },
  };
  return res;
}

/**
 * Run middleware and apply the result to res.
 * Contracts:
 *   - middleware returns null → passthrough, caller must execute itself
 *   - middleware returns {status, body} → apply to res
 */
async function runMiddleware(middleware, req, res, execute) {
  let result = await middleware(req, res, execute);
  if (result === null) result = await execute();
  if (result) {
    res.status(result.status).json(result.body);
  }
  return result;
}

// ------------------------------------------------------------------
// 1. No Idempotency-Key header → passes through (no-op)
// ------------------------------------------------------------------
{
  const store = createIdempotencyStore();
  const middleware = createIdempotencyMiddleware(store);
  let sendCalls = 0;

  const execute = async () => {
    sendCalls += 1;
    return { status: 200, body: { success: true, messageId: 'no-key-1' } };
  };

  const req1 = mockReq('POST', '/send', { chatId: 'x', message: 'hi' }, {});
  const res1 = mockRes();
  const result1 = await runMiddleware(middleware, req1, res1, execute);
  assert.strictEqual(result1.status, 200);
  assert.strictEqual(sendCalls, 1, 'executes when no key present');
  assert.strictEqual(res1._status, 200);

  const req2 = mockReq('POST', '/send', { chatId: 'x', message: 'hi' }, {});
  const res2 = mockRes();
  await runMiddleware(middleware, req2, res2, execute);
  assert.strictEqual(sendCalls, 2, 're-executes when no key — no dedup without key');

  console.log('  ✓ no Idempotency-Key → passes through (no dedup)');
}

// ------------------------------------------------------------------
// 2. Same key+route+payload → cached response returned without re-send
// ------------------------------------------------------------------
{
  const store = createIdempotencyStore();
  const middleware = createIdempotencyMiddleware(store);
  let sendCalls = 0;

  const execute = async () => {
    sendCalls += 1;
    return { status: 200, body: { success: true, messageId: 'dedup-1' } };
  };

  const req1 = mockReq('POST', '/send', { chatId: 'x', message: 'hello' }, {
    'Idempotency-Key': 'key-dedup',
  });
  const res1 = mockRes();
  await runMiddleware(middleware, req1, res1, execute);
  assert.strictEqual(sendCalls, 1);
  assert.strictEqual(res1._status, 200);
  assert.strictEqual(res1._body.messageId, 'dedup-1');

  const req2 = mockReq('POST', '/send', { chatId: 'x', message: 'hello' }, {
    'Idempotency-Key': 'key-dedup',
  });
  const res2 = mockRes();
  await runMiddleware(middleware, req2, res2, execute);
  assert.strictEqual(sendCalls, 1, 'second call with same key+payload does NOT re-send');
  assert.strictEqual(res2._status, 200);
  assert.strictEqual(res2._body.messageId, 'dedup-1', 'returns exact same response body');

  console.log('  ✓ same key+route+payload → cached response, no re-send');
}

// ------------------------------------------------------------------
// 3. Same key, different payload → 409 conflict
// ------------------------------------------------------------------
{
  const store = createIdempotencyStore();
  const middleware = createIdempotencyMiddleware(store);
  let sendCalls = 0;

  const execute = async () => {
    sendCalls += 1;
    return { status: 200, body: { success: true } };
  };

  const req1 = mockReq('POST', '/send', { chatId: 'x', message: 'first' }, {
    'Idempotency-Key': 'key-conflict-mw',
  });
  await runMiddleware(middleware, req1, mockRes(), execute);
  assert.strictEqual(sendCalls, 1);

  const req2 = mockReq('POST', '/send', { chatId: 'x', message: 'second' }, {
    'Idempotency-Key': 'key-conflict-mw',
  });
  const res2 = mockRes();
  await runMiddleware(middleware, req2, res2, execute);
  assert.strictEqual(sendCalls, 1, 'conflicting payload does NOT re-send');
  assert.strictEqual(res2._status, 409, 'conflicting payload returns 409');
  assert.ok(res2._body.error, '409 body includes error message');

  console.log('  ✓ same key, different payload → 409 conflict');
}

// ------------------------------------------------------------------
// 4. Concurrent identical requests → single send (coalescing)
// ------------------------------------------------------------------
{
  const store = createIdempotencyStore();
  const middleware = createIdempotencyMiddleware(store);
  let sendCalls = 0;

  const execute = async () => {
    sendCalls += 1;
    await new Promise(r => setTimeout(r, 20));
    return { status: 200, body: { success: true, messageId: 'coalesced-mw' } };
  };

  const req = () => mockReq('POST', '/send', { chatId: 'x', message: 'concurrent' }, {
    'Idempotency-Key': 'key-concurrent-mw',
  });

  const responses = [
    mockRes(), mockRes(), mockRes(), mockRes(),
  ];
  await Promise.all(responses.map(r => runMiddleware(middleware, req(), r, execute)));

  assert.strictEqual(sendCalls, 1, 'only one send for concurrent identical requests');
  for (const res of responses) {
    assert.strictEqual(res._status, 200);
    assert.strictEqual(res._body.messageId, 'coalesced-mw');
  }

  console.log('  ✓ concurrent identical requests coalesce into a single send');
}

// ------------------------------------------------------------------
// 5. Different routes with same key each execute independently
// ------------------------------------------------------------------
{
  const store = createIdempotencyStore();
  const middleware = createIdempotencyMiddleware(store);
  let sendCalls = 0;

  const execute = async () => {
    sendCalls += 1;
    return { status: 200, body: { success: true, n: sendCalls } };
  };

  const headers = { 'Idempotency-Key': 'key-same-multi-route' };

  await runMiddleware(middleware, mockReq('POST', '/send', { chatId: 'x', message: 'a' }, headers), mockRes(), execute);
  await runMiddleware(middleware, mockReq('POST', '/send-media', { chatId: 'x', filePath: 'y' }, headers), mockRes(), execute);
  assert.strictEqual(sendCalls, 2, 'different routes each execute');

  await runMiddleware(middleware, mockReq('POST', '/send', { chatId: 'x', message: 'a' }, headers), mockRes(), execute);
  assert.strictEqual(sendCalls, 2, 'repeating /send is cached');

  console.log('  ✓ different routes with same key execute independently');
}

// ------------------------------------------------------------------
// 6. GET routes are NOT affected (non-mutating)
// ------------------------------------------------------------------
{
  const store = createIdempotencyStore();
  const middleware = createIdempotencyMiddleware(store);
  let execCalls = 0;

  const execute = async () => {
    execCalls += 1;
    return { status: 200, body: { calls: execCalls } };
  };

  const headers = { 'Idempotency-Key': 'key-get' };

  // GET should always pass through regardless of idempotency
  await runMiddleware(middleware, mockReq('GET', '/health', {}, headers), mockRes(), execute);
  await runMiddleware(middleware, mockReq('GET', '/health', {}, headers), mockRes(), execute);
  assert.strictEqual(execCalls, 2, 'GET routes always re-execute');

  console.log('  ✓ GET routes are not deduped');
}

// ------------------------------------------------------------------
// 7. Non-200 status and body are cached exactly
// ------------------------------------------------------------------
{
  const store = createIdempotencyStore();
  const middleware = createIdempotencyMiddleware(store);
  let sendCalls = 0;

  const execute = async () => {
    sendCalls += 1;
    return { status: 201, body: { success: true, messageId: 'custom-status' } };
  };

  const headers = { 'Idempotency-Key': 'key-status' };
  const req1 = mockReq('POST', '/send', { chatId: 'x', message: 'test' }, headers);
  const res1 = mockRes();
  await runMiddleware(middleware, req1, res1, execute);
  assert.strictEqual(res1._status, 201);
  assert.strictEqual(sendCalls, 1);

  // Repeat — same non-200 status should be cached
  const req2 = mockReq('POST', '/send', { chatId: 'x', message: 'test' }, headers);
  const res2 = mockRes();
  await runMiddleware(middleware, req2, res2, execute);
  assert.strictEqual(sendCalls, 1);
  assert.strictEqual(res2._status, 201, 'non-200 status is cached');
  assert.strictEqual(res2._body.messageId, 'custom-status');

  console.log('  ✓ non-200 status and body are cached exactly');
}

// ------------------------------------------------------------------
// 8. No PII leak via middleware / store
// ------------------------------------------------------------------
{
  const store = createIdempotencyStore();
  const middleware = createIdempotencyMiddleware(store);

  const execute = async () => ({
    status: 200, body: { success: true, messageId: 'pii-test' },
  });

  const headers = { 'Idempotency-Key': 'key-pii-mw' };
  const body = { chatId: '55512345678@s.whatsapp.net', message: 'secret message' };

  await runMiddleware(middleware, mockReq('POST', '/send', body, headers), mockRes(), execute);

  // The stats should not contain any PII
  const stats = store.stats();
  const statsStr = JSON.stringify(stats);
  assert.ok(!statsStr.includes('555'), 'no phone in stats');
  assert.ok(!statsStr.includes('secret'), 'no message content in stats');
  assert.ok(!statsStr.includes('s.whatsapp.net'), 'no chatId in stats');

  console.log('  ✓ no PII in store stats');
}

console.log('\n✅ All idempotency-middleware integration tests passed.');
