import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { once } from 'node:events';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { createServer } from 'node:net';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { test } from 'node:test';
import { fileURLToPath, pathToFileURL } from 'node:url';

const GROUP = '120363001@g.us';
const SENDER = '15550001111@s.whatsapp.net';

async function startBridge(t, { holdVersion = false } = {}) {
  const dir = mkdtempSync(path.join(tmpdir(), 'hermes-bridge-lifecycle-'));
  const bridge = new URL('./bridge.js', import.meta.url);
  const baileys = import.meta.resolve('@whiskeysockets/baileys');
  const stub = path.join(dir, 'socket.mjs');
  const loader = path.join(dir, 'loader.mjs');
  writeFileSync(stub, `
    export * from ${JSON.stringify(baileys)};
    import { EventEmitter } from 'node:events';
    let socket, releaseVersion, releaseMetadata, holdMetadata = false;
    let subject = 'Family', failMetadata = false, metadataCalls = 0;
    export async function useMultiFileAuthState() {
      return { state: { creds: { me: { id: '${SENDER}' } } }, saveCreds() {} };
    }
    export async function fetchLatestBaileysVersion() {
      if (process.env.STUB_HOLD_VERSION) {
        process.send({ type: 'version-pending' });
        await new Promise(resolve => { releaseVersion = resolve; });
      }
      return { version: [2, 3000, 1] };
    }
    export function makeWASocket({ auth }) {
      socket = {
        ev: new EventEmitter(), ws: new EventEmitter(), authState: auth, user: { id: '${SENDER}' },
        async groupMetadata() {
          metadataCalls++;
          if (failMetadata) throw new Error('metadata unavailable');
          const snapshot = subject;
          if (holdMetadata) {
            await new Promise(resolve => { releaseMetadata = resolve; process.send({ type: 'metadata-pending' }); });
          }
          return { subject: snapshot, participants: [] };
        },
        async sendMessage(chatId, message) { return { key: { id: 'sent-poll', remoteJid: chatId }, message }; },
        end() {
          process.send({ type: 'socket-ended' });
          socket.ev.emit('connection.update', { connection: 'close', lastDisconnect: { error: { output: { statusCode: 401 } } } });
        },
        logout() { process.send({ type: 'logged-out' }); },
      };
      setImmediate(() => {
        socket.ev.emit('connection.update', { connection: 'open' });
        process.send({ type: 'ready' });
      });
      return socket;
    }
    process.on('message', async ({ id, event, data, name, fail, release, hold, releaseGroupMetadata }) => {
      if (release) releaseVersion();
      if (hold !== undefined) holdMetadata = hold;
      if (releaseGroupMetadata) { holdMetadata = false; releaseMetadata(); }
      if (name !== undefined) subject = name;
      if (fail !== undefined) failMetadata = fail;
      if (event) for (const listener of socket.ev.listeners(event)) await listener(data);
      process.send({ id, metadataCalls });
    });
  `);
  writeFileSync(loader, `
    export async function resolve(specifier, context, next) {
      if (specifier === '@whiskeysockets/baileys' && context.parentURL === ${JSON.stringify(bridge.href)}) {
        return { url: ${JSON.stringify(pathToFileURL(stub).href)}, shortCircuit: true };
      }
      return next(specifier, context);
    }
  `);
  const listener = createServer().listen(0, '127.0.0.1');
  await once(listener, 'listening');
  const port = listener.address().port;
  await new Promise(resolve => listener.close(resolve));
  const token = 'lifecycle-test-token';
  const child = spawn(process.execPath, ['--no-warnings', '--experimental-loader', pathToFileURL(loader).href,
    fileURLToPath(bridge), '--port', String(port), '--session', path.join(dir, 'session')], {
    windowsHide: true, stdio: ['ignore', 'pipe', 'pipe', 'ipc'],
    env: { PATH: process.env.PATH, SystemRoot: process.env.SystemRoot, TEMP: dir, TMP: dir, HOME: dir,
      HERMES_HOME: dir, HERMES_WHATSAPP_BRIDGE_TOKEN: token,
      STUB_HOLD_VERSION: holdVersion ? '1' : '',
      WHATSAPP_MODE: 'bot', WHATSAPP_ALLOWED_USERS: SENDER, WHATSAPP_GROUP_POLICY: 'open' },
  });
  let output = '';
  const notifications = [];
  const pending = new Map();
  child.stdout.on('data', chunk => { output += chunk; });
  child.stderr.on('data', chunk => { output += chunk; });
  child.on('message', message => {
    notifications.push(message.type);
    pending.get(message.id)?.(message);
    pending.delete(message.id);
  });
  t.after(async () => {
    if (child.exitCode === null && child.signalCode === null) {
      const exited = once(child, 'exit');
      child.kill('SIGKILL');
      await exited;
    }
    rmSync(dir, { recursive: true, force: true });
  });
  await new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(output || 'bridge startup timed out')), 10000);
    child.on('message', message => {
      if (message.type === (holdVersion ? 'version-pending' : 'ready')) { clearTimeout(timer); resolve(); }
    });
    child.once('exit', () => { clearTimeout(timer); reject(new Error(output)); });
  });
  let id = 0;
  return {
    child, notifications,
    waitForOutput(pattern) {
      if (pattern.test(output)) return Promise.resolve();
      return new Promise(resolve => {
        const check = () => { if (pattern.test(output)) { child.stdout.off('data', check); resolve(); } };
        child.stdout.on('data', check);
      });
    },
    command(data) {
      return new Promise(resolve => { pending.set(++id, resolve); child.send({ ...data, id }); });
    },
    async request(route, body) {
      const response = await fetch(`http://127.0.0.1:${port}/${route}`, {
        headers: { authorization: `Bearer ${token}`, 'content-type': 'application/json' },
        method: body ? 'POST' : 'GET', body: body ? JSON.stringify(body) : undefined,
      });
      assert.equal(response.status, 200, output);
      return response.json();
    },
  };
}

const inbound = (id, chatId = GROUP) => ({
  event: 'messages.upsert', data: { type: 'notify', messages: [{
    key: { id, remoteJid: chatId, participant: SENDER, fromMe: false },
    pushName: 'Alice', messageTimestamp: 123, message: { conversation: 'hello' },
  }] },
});

test('shutdown during startup never creates a replacement socket', {
  skip: process.platform === 'win32' && 'Requires POSIX signal delivery', timeout: 15000,
}, async t => {
  const bridge = await startBridge(t, { holdVersion: true });
  const exited = once(bridge.child, 'exit');
  bridge.child.kill('SIGTERM');
  await bridge.waitForOutput(/SIGTERM received/);
  await bridge.command({ release: true });
  assert.deepEqual(await exited, [0, null]);
  assert.ok(!bridge.notifications.includes('ready'));
});

test('inbound group subjects are cached, refreshed on rename, and used for poll votes', { timeout: 15000 }, async t => {
  const bridge = await startBridge(t);
  await bridge.command(inbound('first'));
  assert.equal((await bridge.request('messages'))[0].chatName, 'Family');
  assert.equal((await bridge.command(inbound('cached'))).metadataCalls, 1);
  await bridge.request('messages');
  await bridge.command({ name: 'Friends', event: 'groups.update', data: [{ id: GROUP, subject: 'Friends' }] });
  await bridge.command(inbound('renamed'));
  assert.equal((await bridge.request('messages'))[0].chatName, 'Friends');
  await bridge.request('send-poll', { chatId: GROUP, question: 'Dinner?', options: ['Yes', 'No'] });
  await bridge.command({ event: 'messages.update', data: [{
    key: { id: 'sent-poll', remoteJid: GROUP, participant: SENDER },
    update: { pollUpdates: [{ selectedOptions: ['Yes'], pollUpdateMessageKey: { participant: SENDER } }] },
  }] });
  const votes = await bridge.request('messages');
  assert.equal(votes.length, 1);
  assert.equal(votes[0].chatName, 'Friends');

  await bridge.command({ hold: true, event: 'groups.update', data: [{ id: GROUP }] });
  const paused = once(bridge.child, 'message');
  const pending = bridge.command(inbound('pending'));
  assert.equal((await paused)[0].type, 'metadata-pending');
  await bridge.command({ name: 'Renamed again', event: 'groups.update', data: [{ id: GROUP, subject: 'Renamed again' }] });
  await bridge.command({ releaseGroupMetadata: true });
  await pending;
  await bridge.request('messages');
  await bridge.command(inbound('after-pending-rename'));
  assert.equal((await bridge.request('messages'))[0].chatName, 'Renamed again');
});

test('unavailable group metadata preserves messages and does not change DM names', { timeout: 15000 }, async t => {
  const bridge = await startBridge(t);
  await bridge.command({ fail: true });
  await bridge.command(inbound('fallback'));
  assert.equal((await bridge.request('messages'))[0].chatName, '120363001');
  await bridge.command({ fail: false });
  await bridge.command(inbound('recovered'));
  assert.equal((await bridge.request('messages'))[0].chatName, 'Family');
  const { metadataCalls } = await bridge.command(inbound('dm', SENDER));
  assert.equal((await bridge.request('messages'))[0].chatName, 'Alice');
  assert.equal(metadataCalls, 2);
});

test('signals close the socket without logout or reconnect and exit successfully', {
  skip: process.platform === 'win32' && 'Windows terminates signal targets without POSIX signal handlers', timeout: 15000,
}, async t => {
  for (const signal of ['SIGTERM', 'SIGINT']) {
    await t.test(signal, async t => {
      const bridge = await startBridge(t);
      const exited = once(bridge.child, 'exit');
      bridge.child.kill(signal);
      assert.deepEqual(await exited, [0, null]);
      assert.equal(bridge.notifications.filter(type => type === 'socket-ended').length, 1);
      assert.ok(!bridge.notifications.includes('logged-out'));
      assert.equal(bridge.notifications.filter(type => type === 'ready').length, 1);
    });
  }
});
