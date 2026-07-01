import test from 'node:test';
import assert from 'node:assert/strict';

import { createBridgeAuthMiddleware } from './bridge-auth.js';

function runAuth({ token, path = '/send', authorization } = {}) {
  const middleware = createBridgeAuthMiddleware(token);
  const req = { path, headers: {} };
  if (authorization !== undefined) {
    req.headers.authorization = authorization;
  }

  let statusCode;
  let body;
  let nextCalled = false;
  const res = {
    status(code) {
      statusCode = code;
      return this;
    },
    json(value) {
      body = value;
      return this;
    },
  };

  middleware(req, res, () => {
    nextCalled = true;
  });

  return { statusCode, body, nextCalled };
}

test('bridge auth leaves health public', () => {
  const result = runAuth({ token: 'bridge-token', path: '/health' });
  assert.equal(result.nextCalled, true);
  assert.equal(result.statusCode, undefined);
});

test('bridge auth fails closed when token is unset', () => {
  const result = runAuth({ token: '', path: '/send' });
  assert.equal(result.nextCalled, false);
  assert.equal(result.statusCode, 503);
});

test('bridge auth rejects missing bearer token', () => {
  const result = runAuth({ token: 'bridge-token', path: '/send' });
  assert.equal(result.nextCalled, false);
  assert.equal(result.statusCode, 401);
});

test('bridge auth rejects wrong bearer token without throwing', () => {
  const result = runAuth({
    token: 'bridge-token',
    path: '/send-media',
    authorization: 'Bearer wrong-token',
  });
  assert.equal(result.nextCalled, false);
  assert.equal(result.statusCode, 401);
});

test('bridge auth accepts the configured bearer token', () => {
  const result = runAuth({
    token: 'bridge-token',
    path: '/send',
    authorization: 'Bearer bridge-token',
  });
  assert.equal(result.nextCalled, true);
  assert.equal(result.statusCode, undefined);
});
