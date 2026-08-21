import test from 'node:test';
import assert from 'node:assert/strict';
import { copyFileSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const BRIDGE_DIR = path.dirname(new URL(import.meta.url).pathname);
const FIRST_CONTACT_LID = '267383306489914@lid';
const ALLOWLISTED_PHONE = '19175395595@s.whatsapp.net';
const UNALLOWLISTED_PHONE = '18881234567@s.whatsapp.net';

function writeBridgeHarness(tempDir) {
  for (const file of ['allowlist.js', 'bridge_helpers.js', 'outbound_ids.js', 'owner_message_gate.js']) {
    copyFileSync(path.join(BRIDGE_DIR, file), path.join(tempDir, file));
  }

  const bridgeSource = readFileSync(path.join(BRIDGE_DIR, 'bridge.js'), 'utf8')
    .replace("from '@whiskeysockets/baileys'", "from './test-baileys.mjs'")
    .replace("from 'express'", "from './test-express.mjs'")
    .replace("from '@hapi/boom'", "from './test-boom.mjs'")
    .replace("from 'pino'", "from './test-pino.mjs'")
    .replace("from 'qrcode-terminal'", "from './test-qrcode.mjs'");
  writeFileSync(path.join(tempDir, 'bridge.mjs'), bridgeSource);

  writeFileSync(path.join(tempDir, 'test-baileys.mjs'), `
    export const DisconnectReason = { loggedOut: 401 };
    export const makeWASocket = () => {
      const handlers = new Map();
      const socket = {
        ev: { on: (name, handler) => handlers.set(name, handler) },
        user: { id: '15559998888@s.whatsapp.net' },
        updateMediaMessage: async () => {},
      };
      globalThis.__whatsappTestSocket = socket;
      globalThis.__whatsappTestHandlers = handlers;
      return socket;
    };
    export const useMultiFileAuthState = async () => ({ state: {}, saveCreds: async () => {} });
    export const fetchLatestBaileysVersion = async () => ({ version: [2, 3000, 0] });
    export const downloadMediaMessage = async () => Buffer.from('');
    export const getAggregateVotesInPollMessage = () => [];
    export const decryptPollVote = () => ({});
    export const getKeyAuthor = key => key?.participant || key?.remoteJid || '';
    export const jidNormalizedUser = value => value;
  `);
  writeFileSync(path.join(tempDir, 'test-express.mjs'), `
    const routes = new Map();
    globalThis.__whatsappTestRoutes = routes;
    const app = {
      use: () => {},
      get: (route, handler) => routes.set('GET ' + route, handler),
      post: (route, handler) => routes.set('POST ' + route, handler),
      listen: (_port, _host, callback) => callback(),
    };
    function express() { return app; }
    express.json = () => (_req, _res, next) => next();
    export default express;
  `);
  writeFileSync(path.join(tempDir, 'test-boom.mjs'), `
    export class Boom { constructor() { this.output = { statusCode: 500 }; } }
  `);
  writeFileSync(path.join(tempDir, 'test-pino.mjs'), 'export default function pino() { return {}; }\n');
  writeFileSync(path.join(tempDir, 'test-qrcode.mjs'), 'export default { generate() {} };\n');
}

async function waitForInboundHandler() {
  for (let attempt = 0; attempt < 100; attempt += 1) {
    const handler = globalThis.__whatsappTestHandlers?.get('messages.upsert');
    if (handler) return handler;
    await new Promise(resolve => setTimeout(resolve, 5));
  }
  throw new Error('bridge did not register messages.upsert handler');
}

test('bridge accepts first-contact LID via allowlisted senderPn and emits phone identity', async () => {
  const tempDir = mkdtempSync(path.join(os.tmpdir(), 'hermes-wa-bridge-inbound-'));
  const originalEnv = {
    WHATSAPP_ALLOWED_USERS: process.env.WHATSAPP_ALLOWED_USERS,
    WHATSAPP_DM_POLICY: process.env.WHATSAPP_DM_POLICY,
    WHATSAPP_MODE: process.env.WHATSAPP_MODE,
  };
  const originalArgv = process.argv;

  try {
    writeBridgeHarness(tempDir);
    process.env.WHATSAPP_ALLOWED_USERS = '+19175395595';
    process.env.WHATSAPP_DM_POLICY = 'allowlist';
    process.env.WHATSAPP_MODE = 'bot';
    process.argv = [...process.argv, '--session', tempDir];

    await import(`${pathToFileURL(path.join(tempDir, 'bridge.mjs')).href}?test=${Date.now()}`);
    const inboundHandler = await waitForInboundHandler();

    await inboundHandler({
      type: 'notify',
      messages: [
        {
          key: {
            id: 'allowed-first-contact',
            remoteJid: FIRST_CONTACT_LID,
            senderPn: ALLOWLISTED_PHONE,
            fromMe: false,
          },
          pushName: 'Allowed sender',
          messageTimestamp: 123,
          message: { conversation: 'accepted' },
        },
        {
          key: {
            id: 'blocked-first-contact',
            remoteJid: '188012763865257@lid',
            senderPn: UNALLOWLISTED_PHONE,
            fromMe: false,
          },
          pushName: 'Blocked sender',
          messageTimestamp: 124,
          message: { conversation: 'rejected' },
        },
      ],
    });

    let emitted;
    globalThis.__whatsappTestRoutes.get('GET /messages')(
      {},
      { json: value => { emitted = value; } },
    );

    assert.equal(emitted.length, 1);
    assert.equal(emitted[0].messageId, 'allowed-first-contact');
    assert.equal(emitted[0].body, 'accepted');
    assert.equal(emitted[0].senderId, ALLOWLISTED_PHONE);
    assert.equal(emitted[0].senderName, 'Allowed sender');
  } finally {
    for (const [name, value] of Object.entries(originalEnv)) {
      if (value === undefined) delete process.env[name];
      else process.env[name] = value;
    }
    process.argv = originalArgv;
    delete globalThis.__whatsappTestSocket;
    delete globalThis.__whatsappTestHandlers;
    delete globalThis.__whatsappTestRoutes;
    rmSync(tempDir, { recursive: true, force: true });
  }
});
