import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { once } from 'node:events';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { createServer } from 'node:net';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { test } from 'node:test';
import { fileURLToPath, pathToFileURL } from 'node:url';

const GROUP = '120363001234567890@g.us';
const freePort = () => new Promise((resolve, reject) => {
  const server = createServer().once('error', reject).listen(0, '127.0.0.1', () => {
    const { port } = server.address();
    server.close(() => resolve(port));
  });
});

test('same-account groups reject outbound echoes even when sends time out or have not completed', { timeout: 15000 }, async () => {
  const dir = mkdtempSync(path.join(tmpdir(), 'hermes-group-echo-'));
  const bridge = new URL('./bridge.js', import.meta.url);
  const baileys = import.meta.resolve('@whiskeysockets/baileys');
  const stub = path.join(dir, 'socket.mjs');
  const loader = path.join(dir, 'loader.mjs');
  const token = 'group-echo-test-token';
  let child;
  let output = '';
  try {
    // The real bridge routes and inbound handlers run; only the network transport is replaced.
    writeFileSync(stub, `
      export * from ${JSON.stringify(baileys)};
      import { initAuthCreds } from ${JSON.stringify(baileys)};
      import { EventEmitter } from 'node:events';
      export async function useMultiFileAuthState() {
        const creds = initAuthCreds();
        creds.me = { id: '15551234567:1@s.whatsapp.net' };
        return { state: { creds }, saveCreds() {} };
      }
      export async function fetchLatestBaileysVersion() { return { version: [2, 3000, 1] }; }
      export function makeWASocket({ auth }) {
        const pending = new Map();
        let sequence = 0;
        const socket = {
          ev: new EventEmitter(), ws: new EventEmitter(), authState: auth,
          user: { id: '15551234567:1@s.whatsapp.net' },
          sendMessage(chatId, payload, options) {
            const key = { id: options.messageId || 'OUT-' + ++sequence, remoteJid: chatId, fromMe: true };
            const sent = { key, message: { conversation: payload.text } };
            return new Promise(resolve => {
              pending.set(key.id, () => resolve(sent));
              process.send({ event: 'send-started', id: key.id });
            });
          },
        };
        process.on('message', async command => {
          if (command.action === 'complete') {
            pending.get(command.id)();
            pending.delete(command.id);
            setImmediate(() => process.send({ event: 'completed', id: command.id }));
          } else if (command.action === 'inject') {
            const event = { type: 'notify', messages: [{
              key: { id: command.id, remoteJid: ${JSON.stringify(GROUP)},
                participant: '15551234567@s.whatsapp.net', fromMe: true },
              messageTimestamp: Math.floor(Date.now() / 1000),
              message: { conversation: command.body },
            }] };
            await Promise.all(socket.ev.listeners('messages.upsert').map(listener => listener(event)));
            process.send({ event: 'injected', id: command.id });
          }
        });
        setImmediate(() => {
          socket.ev.emit('connection.update', { connection: 'open' });
          process.send({ event: 'ready' });
        });
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
      fileURLToPath(bridge), '--session', path.join(dir, 'session'), '--port', String(port), '--mode', 'bot'], {
      windowsHide: true,
      stdio: ['ignore', 'pipe', 'pipe', 'ipc'],
      env: { PATH: process.env.PATH, SystemRoot: process.env.SystemRoot, HOME: dir, TEMP: dir, TMP: dir,
        HERMES_HOME: dir, HERMES_WHATSAPP_BRIDGE_TOKEN: token,
        HERMES_IMAGE_CACHE_DIR: path.join(dir, 'images'), HERMES_AUDIO_CACHE_DIR: path.join(dir, 'audio'),
        HERMES_DOCUMENT_CACHE_DIR: path.join(dir, 'documents'),
        WHATSAPP_PROCESS_FROM_ME_GROUPS: 'true', WHATSAPP_GROUP_POLICY: 'allowlist',
        WHATSAPP_GROUP_ALLOWED_USERS: GROUP, WHATSAPP_DM_POLICY: 'disabled', WHATSAPP_SEND_TIMEOUT_MS: '50' },
    });
    child.stdout.setEncoding('utf8').on('data', chunk => { output += chunk; });
    child.stderr.setEncoding('utf8').on('data', chunk => { output += chunk; });
    const nextMessage = () => once(child, 'message', { signal: AbortSignal.timeout(10000) }).then(([message]) => message);
    assert.equal((await nextMessage()).event, 'ready', output);
    const request = (method, route, body) => fetch(`http://127.0.0.1:${port}${route}`, {
      method, headers: { authorization: ['Bearer', token].join(' '), 'content-type': 'application/json' },
      body: body ? JSON.stringify(body) : undefined,
    });
    const command = async payload => {
      const ack = nextMessage();
      child.send(payload);
      const response = await ack;
      assert.equal(response.id, payload.id, output);
    };
    // A same-account message not sent by Hermes still reaches intake without owner-DM handover.
    await command({ action: 'inject', id: 'OWNER-TYPED', body: '@hermes status' });
    const inbound = await (await request('GET', '/messages')).json();
    assert.equal(inbound.length, 1, output);
    assert.equal(inbound[0].body, '@hermes status');
    assert.equal(inbound[0].fromOwner, false);

    for (const phase of ['after-completion', 'before-completion']) {
      const started = nextMessage();
      const sending = request('POST', '/send', { chatId: GROUP, message: 'agent reply' });
      const { id, event } = await started;
      assert.equal(event, 'send-started', output);
      const response = await sending;
      assert.equal(response.status, 500);
      assert.match((await response.json()).error, /timed out/);
      if (phase === 'after-completion') await command({ action: 'complete', id });
      await command({ action: 'inject', id, body: 'agent reply' });
      assert.deepEqual(await (await request('GET', '/messages')).json(), [], `${phase}: an outbound echo reached intake`);
      if (phase === 'before-completion') await command({ action: 'complete', id });
    }
  } finally {
    if (child && child.exitCode === null && child.signalCode === null) {
      const exited = once(child, 'exit');
      child.kill();
      await exited;
    }
    assert.equal(path.dirname(dir), tmpdir());
    rmSync(dir, { recursive: true, force: true });
  }
});
