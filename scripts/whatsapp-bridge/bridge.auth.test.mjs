import assert from 'node:assert/strict';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { createServer } from 'node:net';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { spawn } from 'node:child_process';
import { pathToFileURL, fileURLToPath } from 'node:url';
import { test } from 'node:test';

import { bridgeAuthProof } from './bridge_auth.js';

const freePort = () => new Promise((resolve, reject) => {
  const server = createServer().once('error', reject).listen(0, '127.0.0.1', () => {
    const { port } = server.address();
    server.close(() => resolve(port));
  });
});

const CHAT = '15550001111@s.whatsapp.net';
// Every route bridge.js serves besides /auth-proof. A new route must be added here and stay behind the gate.
const ROUTES = [
  ['GET', '/messages'],
  ['GET', '/health'],
  ['GET', `/chat/${CHAT}`],
  ['POST', '/send', { chatId: CHAT, message: 'hi' }],
  ['POST', '/edit', { chatId: CHAT, messageId: 'M1', message: 'hi' }],
  ['POST', '/send-media', { chatId: CHAT, filePath: '/nonexistent' }],
  ['POST', '/send-poll', { chatId: CHAT, question: 'q', options: ['a', 'b'] }],
  ['POST', '/send-location', { chatId: CHAT, latitude: 1, longitude: 2 }],
  ['POST', '/typing', { chatId: CHAT }],
  ['POST', '/read', { chatId: CHAT, messageId: 'M1' }],
];

test('bridge.js serves /auth-proof openly and puts every other route behind the session bearer', async () => {
  const dir = mkdtempSync(path.join(tmpdir(), 'hermes-bridge-auth-'));
  const bridge = new URL('./bridge.js', import.meta.url);
  const baileys = import.meta.resolve('@whiskeysockets/baileys');
  const stub = path.join(dir, 'socket.mjs');
  const loader = path.join(dir, 'loader.mjs');
  const sentFile = path.join(dir, 'sent.log');
  const token = 'bridge-auth-test-token';
  let child;
  try {
    // Baileys stubbed as in bridge.passkey.test.mjs: a paired session that opens at once; sends are recorded, never sent.
    writeFileSync(stub, `
      export * from ${JSON.stringify(baileys)};
      import { initAuthCreds } from ${JSON.stringify(baileys)};
      import { EventEmitter } from 'node:events';
      import { appendFileSync } from 'node:fs';
      export async function useMultiFileAuthState() {
        const creds = initAuthCreds();
        creds.me = { id: '15551234567:1@s.whatsapp.net' };
        return { state: { creds }, saveCreds() {} };
      }
      export async function fetchLatestBaileysVersion() { return { version: [2, 3000, 1] }; }
      export function makeWASocket({ auth }) {
        const record = (kind) => appendFileSync(${JSON.stringify(sentFile)}, kind + '\\n');
        const socket = {
          ev: new EventEmitter(), ws: new EventEmitter(), authState: auth,
          user: { id: '15551234567:1@s.whatsapp.net' },
          async sendMessage(jid) { record('send'); return { key: { id: 'SENT-' + jid } }; },
          async sendPresenceUpdate() { record('presence'); },
          async readMessages() { record('read'); },
          async groupMetadata() { record('group'); return { subject: 'g', participants: [] }; },
        };
        setImmediate(() => socket.ev.emit('connection.update', { connection: 'open' }));
        return socket;
      }
    `);
    writeFileSync(loader, `
      export async function resolve(specifier, context, next) {
        if (specifier === '@whiskeysockets/baileys' && context.parentURL === ${JSON.stringify(bridge.href)}) {
          return { url: ${JSON.stringify(pathToFileURL(stub).href)}, shortCircuit: true };
        }
        return next(specifier, context);
      }
    `);
    const port = await freePort();
    child = spawn(process.execPath, ['--no-warnings', '--experimental-loader', pathToFileURL(loader).href,
      fileURLToPath(bridge), '--session', path.join(dir, 'session'), '--port', String(port)], {
      windowsHide: true,
      env: { PATH: process.env.PATH, SystemRoot: process.env.SystemRoot, TEMP: dir, TMP: dir, HERMES_HOME: dir,
        HERMES_WHATSAPP_BRIDGE_TOKEN: token },
    });
    let output = '';
    child.stdout.setEncoding('utf8').on('data', chunk => { output += chunk; });
    child.stderr.setEncoding('utf8').on('data', chunk => { output += chunk; });
    const base = `http://127.0.0.1:${port}`;
    const request = (method, route, body, headers = {}) => fetch(base + route, {
      method,
      headers: body ? { ...headers, 'content-type': 'application/json' } : headers,
      body: body ? JSON.stringify(body) : undefined,
    });

    const challenge = { 'x-hermes-bridge-challenge': 'nonce-1' };
    let proof;
    for (let attempt = 0; attempt < 100 && !proof; attempt++) {
      assert.equal(child.exitCode, null, output);
      proof = await request('GET', '/auth-proof', undefined, challenge).catch(() => null);
      if (!proof) await new Promise(resolve => setTimeout(resolve, 100));
    }
    assert.ok(proof, output);
    assert.equal(proof.status, 200, output);
    assert.deepEqual(await proof.json(), { authProof: bridgeAuthProof(token, 'nonce-1') });

    const rejected = [{}, { authorization: 'Bearer wrong' }, { authorization: token }, { authorization: `Basic ${token}` }];
    for (const headers of rejected) {
      for (const [method, route, body] of [...ROUTES, ['GET', '/not-a-route']]) {
        const response = await request(method, route, body, headers);
        assert.equal(response.status, 401, `${method} ${route} with ${JSON.stringify(headers)}`);
      }
    }
    // None of the rejected calls may have reached the socket.
    assert.throws(() => readFileSync(sentFile), { code: 'ENOENT' });

    const bearer = { authorization: `Bearer ${token}` };
    assert.equal((await request('GET', '/messages', undefined, bearer)).status, 200);
    assert.equal((await request('GET', '/health', undefined, bearer)).status, 200);
    assert.equal((await request('POST', '/send', ROUTES[3][2], bearer)).status, 200, output);
    assert.match(readFileSync(sentFile, 'utf8'), /send/);
  } finally {
    if (child && child.exitCode === null) {
      const exited = new Promise(resolve => child.once('exit', resolve));
      child.kill();
      await exited;
    }
    assert.equal(path.dirname(dir), tmpdir());
    rmSync(dir, { recursive: true, force: true });
  }
});
