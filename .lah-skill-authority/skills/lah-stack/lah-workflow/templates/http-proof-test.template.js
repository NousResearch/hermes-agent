/**
 * http-route-proof.template.js
 *
 * Starter template for HTTP route proof missions.
 * Copy this file and rename for your route under test.
 *
 * Covers:
 *   - buildServer() with createApp() + ephemeral port
 *   - req() helper using http.request() (avoids http.get header trap)
 *   - env isolation per test run
 *   - auth proof skeleton
 *   - CRUD route proof skeleton
 */

import test from 'node:test';
import assert from 'node:assert/strict';
import { request } from 'node:http';
import { writeFileSync, mkdirSync, rmSync, readFileSync, statSync } from 'node:fs';
import { join } from 'node:path';
import { randomUUID, createHash } from 'node:crypto';

// ── Configuration ──────────────────────────────────────────────────────
// Set these to match your route's auth scheme
const ADMIN_KEY = 'route-proof-test-admin-key';
const BRIDGE_KEY = 'route-proof-test-bridge-key';
const BASE_PATH = '/creative-production/draw-things';  // change per route

// ── Test harness ───────────────────────────────────────────────────────

/**
 * Set temp env vars for isolated test run. Returns { tmp, cleanup() }.
 */
function setupRun() {
  const id = randomUUID().slice(0, 8);
  const tmp = `/tmp/route-proof-${id}`;
  mkdirSync(tmp, { recursive: true });

  // Save originals
  const saved = {};
  function set(k, v) {
    saved[k] = process.env[k];
    process.env[k] = v;
  }

  set('ADMIN_API_KEY', ADMIN_KEY);
  set('CLOE_BRIDGE_API_KEY', BRIDGE_KEY);
  set('DRAW_THINGS_JOB_STORE_FILE', join(tmp, 'store.json'));
  set('DRAW_THINGS_ASSETS_DIR', join(tmp, 'assets'));
  set('NODE_ENV', 'test');
  set('ALLOWED_ORIGIN', 'http://localhost');

  return {
    tmp,
    cleanup() {
      for (const [k, v] of Object.entries(saved)) {
        if (v === undefined) delete process.env[k];
        else process.env[k] = v;
      }
      try { rmSync(tmp, { recursive: true, force: true }); } catch {}
    },
  };
}

/**
 * Build an Express app server on ephemeral 127.0.0.1:0.
 * @returns {Promise<{url: Function, close: Function}>}
 */
async function buildServer() {
  const { createApp } = await import('../src/server.js');
  const app = createApp();
  return new Promise((resolve) => {
    const srv = app.listen(0, '127.0.0.1', () => {
      const port = srv.address().port;
      resolve({
        url: (path) => `http://127.0.0.1:${port}${path}`,
        close: () => { srv.close(); },
      });
    });
  });
}

/**
 * Issue an HTTP request and parse JSON response.
 *
 * IMPORTANT: Uses http.request() with {hostname, port, path, method, headers}
 * — NOT http.get(urlString, options) which may silently drop headers.
 */
async function req(method, url, body, headers) {
  const u = new URL(url);
  const opts = {
    hostname: '127.0.0.1',
    port: u.port,
    path: u.pathname,
    method,
    headers: headers || {},
  };
  if (body) opts.headers['Content-Type'] = 'application/json';

  return new Promise((resolve) => {
    const r = request(opts, (res) => {
      let d = '';
      res.on('data', (c) => (d += c));
      res.on('end', () => {
        let b;
        try { b = JSON.parse(d); } catch { b = d; }
        resolve({ status: res.statusCode, headers: res.headers, body: b });
      });
    });
    r.on('error', (e) => resolve({ status: 0, body: { error: e.message } }));
    if (body) r.write(JSON.stringify(body));
    r.end();
  });
}

// ── Common header presets ──────────────────────────────────────────────

const AUTH_ADMIN   = { 'x-admin-api-key': ADMIN_KEY };
const AUTH_BRIDGE  = { 'x-cloe-bridge-api-key': BRIDGE_KEY };
const AUTH_WRONG   = { 'x-admin-api-key': 'wrong-key' };
const NO_AUTH      = {};

// ═════════════════════════════════════════════════════════════════════════
//  TESTS — replace with your route's contract
// ═════════════════════════════════════════════════════════════════════════

test('Example HTTP route proof', async (t) => {
  const env = setupRun();
  const svc = await buildServer();

  // ── Auth proof ────────────────────────────────────────────────────
  await t.test('no auth → 401', async () => {
    const r = await req('GET', svc.url(`${BASE_PATH}/status`), null, NO_AUTH);
    assert.equal(r.status, 401);
  });

  await t.test('wrong key → 401', async () => {
    const r = await req('GET', svc.url(`${BASE_PATH}/status`), null, AUTH_WRONG);
    assert.equal(r.status, 401);
  });

  await t.test('valid key → 200', async () => {
    const r = await req('GET', svc.url(`${BASE_PATH}/status`), null, AUTH_ADMIN);
    assert.equal(r.status, 200);
  });

  // ── Route contract ────────────────────────────────────────────────
  await t.test('POST valid body → 201', async () => {
    const r = await req('POST', svc.url(`${BASE_PATH}/job`), {
      job_id: `test_${Date.now()}`,
      prompt: 'A valid test prompt',
    }, AUTH_ADMIN);
    assert.equal(r.status, 201);
    assert.equal(r.body.ok, true);
  });

  await t.test('POST invalid body → 400', async () => {
    const r = await req('POST', svc.url(`${BASE_PATH}/job`), {
      job_id: `test_${Date.now()}`,
      // missing required prompt field
    }, AUTH_ADMIN);
    assert.equal(r.status, 400);
  });

  // ── Health endpoint ───────────────────────────────────────────────
  await t.test('GET /health still works (regression)', async () => {
    const r = await req('GET', svc.url('/health'));
    assert.equal(r.status, 200);
    assert.equal(r.body.ok, true);
  });

  // ── Cleanup ───────────────────────────────────────────────────────
  svc.close();
  env.cleanup();
});
