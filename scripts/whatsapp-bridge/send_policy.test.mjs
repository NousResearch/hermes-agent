/**
 * Behaviour tests for the send-policy guard (config-gated read-only mode).
 *
 * Part 1 (pure): createSendPolicyGuard policy resolution + fail-closed parsing.
 * Part 2 (live): an express server wired with the guard — outbound routes 403 with
 * recon rows, observation routes pass, /health reports the policy. A real server is
 * the bite test for the wiring: a guard that is imported but never `app.use`d fails
 * these round-trips.
 */

import { strict as assert } from 'node:assert';
import express from 'express';
import http from 'node:http';
import { mkdtempSync, readFileSync, existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';

import { OUTBOUND_ROUTES, createSendPolicyGuard } from './bridge_helpers.js';

function fetchJson(url, options = {}) {
  return new Promise((resolve, reject) => {
    const req = http.request(url, { method: options.method || 'GET' }, (res) => {
      let body = '';
      res.on('data', (c) => (body += c));
      res.on('end', () => {
        try {
          resolve({ status: res.statusCode, body: body ? JSON.parse(body) : null });
        } catch (e) {
          resolve({ status: res.statusCode, body });
        }
      });
    });
    req.on('error', reject);
    if (options.body) req.write(options.body);
    req.end();
  });
}

const tick = () => new Promise((resolve) => setImmediate(resolve));

// -- Part 1: policy resolution (pure) --------------------------------------

// default and 'open' never disable
{
  assert.equal(createSendPolicyGuard({}).sendsDisabled, false);
  assert.equal(createSendPolicyGuard({ sendPolicy: 'open' }).sendsDisabled, false);
  assert.equal(createSendPolicyGuard({ sendPolicy: 'OPEN' }).sendsDisabled, false);
  console.log('  ✓ open/default policy never disables sends');
}

// any other value fails CLOSED (a typo must never silently enable sends)
{
  for (const value of ['disabled', 'DISABLED', 'read-only', 'typo', '']) {
    const guard = createSendPolicyGuard({ sendPolicy: value });
    assert.equal(guard.sendsDisabled, true, value);
    assert.equal(guard.policy, String(value).trim().toLowerCase(), value);
  }
  // unset (absent or null) falls back to the default policy — the same semantics as the
  // adapter's extra.get("send_policy", "open"); only PROVIDED non-open strings fail closed.
  assert.equal(createSendPolicyGuard({}).sendsDisabled, false);
  assert.equal(createSendPolicyGuard({ sendPolicy: null }).sendsDisabled, false);
  console.log('  ✓ every provided non-open value fails closed; unset = default open');
}

// recon rows only written when disabled and a logPath is set
{
  const dir = mkdtempSync(path.join(tmpdir(), 'send-policy-'));
  const logPath = path.join(dir, 'recon', 'send_policy_recon.jsonl');
  const guard = createSendPolicyGuard({ sendPolicy: 'disabled', logPath });

  const req = { path: '/send' };
  const res = {
    statusCode: 0,
    status(code) { this.statusCode = code; return this; },
    json(body) { this.body = body; return this; },
  };
  let nextCalled = false;
  guard(req, res, () => { nextCalled = true; });

  assert.equal(nextCalled, false);
  assert.equal(res.statusCode, 403);
  assert.match(res.body.error, /read_only/);
  await tick();
  assert.ok(existsSync(logPath), 'refusal must append a recon row');
  const row = JSON.parse(readFileSync(logPath, 'utf8').trim().split('\n').pop());
  assert.equal(row.route, '/send');
  assert.equal(row.reason, 'send_policy_disabled');
  assert.equal(typeof row.ts, 'number');
  console.log('  ✓ refusal writes a JSONL recon row (ts, route, reason)');
}

// observation routes and open policy pass through to next()
{
  const guard = createSendPolicyGuard({ sendPolicy: 'disabled' });
  for (const p of ['/messages', '/chat/15550000001', '/health']) {
    let next = false;
    guard({ path: p }, {}, () => { next = true; });
    assert.equal(next, true, p);
  }
  const openGuard = createSendPolicyGuard({ sendPolicy: 'open' });
  let openNext = false;
  openGuard({ path: '/send' }, {}, () => { openNext = true; });
  assert.equal(openNext, true, 'open policy must pass outbound routes through');
  console.log('  ✓ observation routes and open policy pass through');
}

// the route inventory covers every outbound bridge route
{
  for (const r of ['/send', '/edit', '/send-media', '/send-poll', '/send-location', '/typing', '/read']) {
    assert.ok(OUTBOUND_ROUTES.has(r), r);
  }
  assert.equal(OUTBOUND_ROUTES.size, 7);
  console.log('  ✓ OUTBOUND_ROUTES covers the 7 outbound routes');
}

// -- Part 2: live express wiring (the real bite test) -----------------------

async function withServer(sendPolicy, fn) {
  const app = express();
  app.use(express.json());
  const guard = createSendPolicyGuard({
    sendPolicy,
    logPath: path.join(tmpdir(), `send-policy-live-${Date.now()}.jsonl`),
  });
  app.use(guard);

  app.post('/send', (req, res) => res.json({ success: true, hit: 'route' }));
  app.get('/health', (req, res) => res.json({ status: 'connected', sendsDisabled: guard.sendsDisabled, sendPolicy: guard.policy }));

  const server = app.listen(0, '127.0.0.1');
  await new Promise((resolve) => server.once('listening', resolve));
  const port = server.address().port;
  try {
    return await fn(`http://127.0.0.1:${port}`);
  } finally {
    server.close();
    await new Promise((resolve) => server.close(resolve));
  }
}

// disabled: all 7 outbound routes 403, observation route passes, health reports policy
await withServer('disabled', async (base) => {
  for (const route of OUTBOUND_ROUTES) {
    const res = await fetchJson(`${base}${route}`, { method: 'POST', body: '{}' });
    assert.equal(res.status, 403, route);
    assert.match(res.body.error, /read_only/, route);
  }
  const health = await fetchJson(`${base}/health`);
  assert.equal(health.status, 200);
  assert.equal(health.body.sendsDisabled, true);
  assert.equal(health.body.sendPolicy, 'disabled');
  console.log('  ✓ live server: 7/7 outbound routes 403, /health reports the policy');
});

// open (default): outbound route reaches its handler (control — proves the test can fail)
await withServer('open', async (base) => {
  const res = await fetchJson(`${base}/send`, { method: 'POST', body: '{}' });
  assert.equal(res.status, 200);
  assert.equal(res.body.hit, 'route');
  console.log('  ✓ live server: open policy reaches the route (control)');
});

console.log('\nsend-policy guard tests passed');
