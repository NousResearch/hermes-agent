import { strict as assert } from 'node:assert';
import { once } from 'node:events';
import http from 'node:http';

import { createBridgeApp } from './http_app.js';

function request(server, host) {
  const address = server.address();
  return new Promise((resolve, reject) => {
    const req = http.request({
      hostname: '127.0.0.1',
      port: address.port,
      path: '/health',
      headers: { Host: host },
    }, (res) => {
      let body = '';
      res.setEncoding('utf8');
      res.on('data', (chunk) => { body += chunk; });
      res.on('end', () => resolve({ body, headers: res.headers, status: res.statusCode }));
    });
    req.on('error', reject);
    req.end();
  });
}

const app = createBridgeApp();
app.get('/health', (req, res) => res.json({ status: 'ok' }));

const server = app.listen(0, '127.0.0.1');
await once(server, 'listening');

try {
  const accepted = await request(server, '127.0.0.1');
  assert.equal(accepted.status, 200);
  assert.deepEqual(JSON.parse(accepted.body), { status: 'ok' });
  assert.equal(accepted.headers['x-powered-by'], undefined);

  const rejected = await request(server, 'example.com');
  assert.equal(rejected.status, 400);
  assert.deepEqual(JSON.parse(rejected.body), {
    error: 'Invalid Host header. Bridge accepts loopback hosts only.',
  });
  assert.equal(rejected.headers['x-powered-by'], undefined);
} finally {
  server.close();
  await once(server, 'close');
}
