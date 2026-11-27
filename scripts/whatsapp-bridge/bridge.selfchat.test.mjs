import assert from 'node:assert/strict';
import { mkdtempSync, rmSync } from 'node:fs';
import { registerHooks } from 'node:module';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { after, before, test } from 'node:test';

const handlers = new Map();
const routes = new Map();
const ready = Promise.withResolvers();
const socket = {
  user: {},
  ev: {
    on(name, handler) {
      handlers.set(name, handler);
      if (name === 'messages.upsert') ready.resolve();
    },
  },
};
const fixtureKey = Symbol.for('hermes.whatsapp.selfchat.test');
const originalEnv = Object.fromEntries(
  ['HOME', 'HERMES_HOME', 'WHATSAPP_MODE'].map(key => [key, process.env[key]]),
);
let home;
let hooks;

before(async () => {
  home = mkdtempSync(path.join(tmpdir(), 'hermes-whatsapp-selfchat-'));
  Object.assign(process.env, { HOME: home, HERMES_HOME: home, WHATSAPP_MODE: 'self-chat' });
  globalThis[fixtureKey] = { socket, routes };

  // Replace external transports only: run the real bridge intake and /messages
  // handler without a WhatsApp connection, HTTP listener, or installed credentials.
  const fixture = 'const fixture = globalThis[Symbol.for("hermes.whatsapp.selfchat.test")];';
  const stubs = new Map([
    ['@whiskeysockets/baileys', `${fixture}
      export const makeWASocket = () => fixture.socket;
      export const useMultiFileAuthState = async () => ({ state: {}, saveCreds() {} });
      export const fetchLatestBaileysVersion = async () => ({ version: [2, 3000, 0] });
      export const DisconnectReason = {};
      export const jidNormalizedUser = id => id;
      export function downloadMediaMessage() { throw new Error('Unexpected media download'); }
      export function getAggregateVotesInPollMessage() { throw new Error('Unexpected poll'); }
      export function decryptPollVote() { throw new Error('Unexpected poll'); }
      export function getKeyAuthor() { throw new Error('Unexpected poll'); }
    `],
    ['express', `${fixture}
      function express() {
        return {
          use() {},
          get(route, handler) { fixture.routes.set(route, handler); },
          post() {},
          listen(port, host, callback) { callback(); },
        };
      }
      express.json = () => () => {};
      export default express;
    `],
    ['@hapi/boom', 'export class Boom extends Error {}'],
    ['pino', 'export default () => ({})'],
    ['qrcode-terminal', 'export default {}'],
  ]);
  hooks = registerHooks({
    resolve(specifier, context, nextResolve) {
      if (stubs.has(specifier)) {
        return { url: `data:text/javascript,${encodeURIComponent(stubs.get(specifier))}`, shortCircuit: true };
      }
      return nextResolve(specifier, context);
    },
  });
  await import('./bridge.js');
  await ready.promise;
});

after(() => {
  hooks?.deregister();
  delete globalThis[fixtureKey];
  for (const [key, value] of Object.entries(originalEnv)) {
    if (value === undefined) delete process.env[key];
    else process.env[key] = value;
  }
  if (home) rmSync(home, { recursive: true, force: true });
});

const pn = '15551234567';
const lid = '67427329167522';
for (const type of ['notify', 'append']) {
  for (const [name, accountLid, chatId, accepted] of [
    ['LID self-chat', `${lid}:10@lid`, `${lid}@lid`, true],
    // Meta AI prompts use the legacy PN identity even when the account has a LID.
    ['Meta AI on PN with LID', `${lid}:10@lid`, `${pn}@s.whatsapp.net`, false],
    ['legacy PN self-chat without LID', undefined, `${pn}@s.whatsapp.net`, true],
    ['legacy PN self-chat with empty LID', '', `${pn}@s.whatsapp.net`, true],
    ['third-party PN with LID', `${lid}:10@lid`, '15559876543@s.whatsapp.net', false],
    ['third-party LID', `${lid}:10@lid`, '88888888888888@lid', false],
    ['third-party PN without LID', undefined, '15559876543@s.whatsapp.net', false],
  ]) {
    test(`${type}: ${name} ${accepted ? 'reaches' : 'never reaches'} the gateway queue`, async () => {
      socket.user = { id: `${pn}:10@s.whatsapp.net`, lid: accountLid };
      const messageId = `${type}-${name}`;
      await handlers.get('messages.upsert')({
        type,
        messages: [{
          key: { id: messageId, remoteJid: chatId, fromMe: true },
          message: { conversation: 'Please handle this request' },
          messageTimestamp: 123,
        }],
      });

      let queued;
      routes.get('/messages')({}, { json: messages => { queued = messages; } });
      assert.deepEqual(queued.map(event => event.messageId), accepted ? [messageId] : []);
      if (accepted) {
        assert.equal(queued[0].chatId, chatId);
        assert.equal(queued[0].body, 'Please handle this request');
      }
    });
  }
}
