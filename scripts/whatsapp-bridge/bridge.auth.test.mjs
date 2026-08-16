import assert from 'node:assert/strict';
import http from 'node:http';
import { spawnSync } from 'node:child_process';
import test from 'node:test';
import express from 'express';

import {
  createAuthenticatedBridgeApp,
  createBridgeAuthMiddleware,
  validateBridgeLaunch,
} from './bridge_helpers.js';

function responseRecorder() {
  return {
    statusCode: 200,
    headers: {},
    body: null,
    status(code) {
      this.statusCode = code;
      return this;
    },
    set(name, value) {
      this.headers[name] = value;
      return this;
    },
    json(body) {
      this.body = body;
      return this;
    },
  };
}

function invokeAuth(authorization, expectedToken = 'session-token') {
  const req = { headers: {} };
  if (authorization !== undefined) req.headers.authorization = authorization;
  const res = responseRecorder();
  let nextCalled = false;
  createBridgeAuthMiddleware(expectedToken)(req, res, () => {
    nextCalled = true;
  });
  return { res, nextCalled };
}

test('bridge bearer auth rejects requests without credentials', () => {
  const { res, nextCalled } = invokeAuth(undefined);

  assert.equal(nextCalled, false);
  assert.equal(res.statusCode, 401);
  assert.equal(res.headers['WWW-Authenticate'], 'Bearer');
  assert.deepEqual(res.body, { error: 'Unauthorized' });
});

test('bridge bearer auth rejects malformed and incorrect credentials', () => {
  for (const authorization of [
    '',
    'Basic session-token',
    'Bearer',
    'Bearer wrong-token',
    'Bearer session-token extra',
  ]) {
    const { res, nextCalled } = invokeAuth(authorization);
    assert.equal(nextCalled, false, authorization);
    assert.equal(res.statusCode, 401, authorization);
  }
});

test('bridge bearer auth accepts the exact session token', () => {
  const { res, nextCalled } = invokeAuth('Bearer session-token');

  assert.equal(nextCalled, true);
  assert.equal(res.statusCode, 200);
  assert.equal(res.body, null);
});

test('bridge bearer auth fails closed without a configured token', () => {
  const { res, nextCalled } = invokeAuth('Bearer session-token', '');

  assert.equal(nextCalled, false);
  assert.equal(res.statusCode, 401);
});

function request(server, { authorization, body }) {
  const { port } = server.address();
  return new Promise((resolve, reject) => {
    const headers = {
      host: 'localhost',
      'content-type': 'application/json',
      'content-length': Buffer.byteLength(body),
    };
    if (authorization) headers.authorization = authorization;
    const req = http.request({
      host: '127.0.0.1', port, path: '/probe', method: 'POST', headers,
    }, (res) => {
      const chunks = [];
      res.on('data', (chunk) => chunks.push(chunk));
      res.on('end', () => resolve({
        status: res.statusCode,
        body: Buffer.concat(chunks).toString('utf8'),
      }));
    });
    req.on('error', reject);
    req.end(body);
  });
}

test('real Express app authenticates before parsing and routes', async (t) => {
  const app = createAuthenticatedBridgeApp(express, 'session-token');
  let routeCalls = 0;
  app.post('/probe', (req, res) => {
    routeCalls += 1;
    res.json({ received: req.body.message });
  });
  app.use((err, _req, res, _next) => {
    res.status(err.status || 500).json({ error: 'Invalid request body' });
  });
  const server = app.listen(0, '127.0.0.1');
  await new Promise((resolve) => server.once('listening', resolve));
  t.after(() => new Promise((resolve) => server.close(resolve)));

  const unauthenticated = await request(server, {
    // Express defaults to a 100 KiB JSON limit. Keep this large enough to prove
    // auth runs before parsing without overflowing local socket buffers.
    body: '{'.repeat(128 * 1024),
  });
  assert.equal(unauthenticated.status, 401);
  assert.equal(routeCalls, 0);

  const malformedAuthenticated = await request(server, {
    authorization: 'Bearer session-token',
    body: '{',
  });
  assert.equal(malformedAuthenticated.status, 400);
  assert.equal(routeCalls, 0);

  const authenticated = await request(server, {
    authorization: 'Bearer session-token',
    body: JSON.stringify({ message: 'private' }),
  });
  assert.equal(authenticated.status, 200);
  assert.deepEqual(JSON.parse(authenticated.body), { received: 'private' });
  assert.equal(routeCalls, 1);
});

test('native Windows serving fails before named-pipe listen', () => {
  assert.throws(
    () => validateBridgeLaunch({
      platform: 'win32',
      pairOnly: false,
      endpoint: String.raw`\\.\pipe\observed-name`,
      token: 'session-token',
    }),
    /Native Windows.*unsupported/,
  );
  assert.doesNotThrow(() => validateBridgeLaunch({
    platform: 'win32',
    pairOnly: true,
    endpoint: '',
    token: '',
  }));
});

test('public npm-start launcher gives actionable gateway guidance', () => {
  const result = spawnSync(process.execPath, ['bridge-launcher.js'], {
    cwd: import.meta.dirname,
    encoding: 'utf8',
  });
  assert.equal(result.status, 1);
  assert.match(result.stderr, /hermes gateway/);
  assert.match(result.stderr, /hermes whatsapp/);
});
