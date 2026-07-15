import assert from 'node:assert/strict';
import { mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import test from 'node:test';

import { createBridgeAuthMiddleware, loadBridgeToken } from './bridge_auth.js';

function responseRecorder() {
  return {
    statusCode: 200,
    payload: null,
    status(code) { this.statusCode = code; return this; },
    json(payload) { this.payload = payload; return this; },
  };
}

test('loads token and accepts exact bearer token', () => {
  const dir = mkdtempSync(path.join(tmpdir(), 'wa-auth-'));
  const file = path.join(dir, 'token');
  writeFileSync(file, 'top-secret\n', 'utf8');
  const token = loadBridgeToken({ tokenPath: file, envToken: '' });
  assert.equal(token, 'top-secret');

  const middleware = createBridgeAuthMiddleware(token);
  const res = responseRecorder();
  let called = false;
  middleware({ get: () => 'Bearer top-secret' }, res, () => { called = true; });
  assert.equal(called, true);
  assert.equal(res.statusCode, 200);
});

test('rejects missing and wrong tokens without leaking expected token', () => {
  const middleware = createBridgeAuthMiddleware('top-secret');
  for (const header of [undefined, '', 'Bearer wrong']) {
    const res = responseRecorder();
    let called = false;
    middleware({ get: () => header }, res, () => { called = true; });
    assert.equal(called, false);
    assert.equal(res.statusCode, 401);
    assert.deepEqual(res.payload, { error: 'Unauthorized' });
  }
});

test('fails closed when no token is configured', () => {
  assert.throws(() => loadBridgeToken({ tokenPath: 'Z:/missing-token', envToken: '' }));
  assert.throws(() => createBridgeAuthMiddleware(''));
});
