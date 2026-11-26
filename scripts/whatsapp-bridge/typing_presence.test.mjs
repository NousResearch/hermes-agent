/**
 * The typing indicator the /typing endpoint forwards to WhatsApp: shown, and withdrawn again.
 */

import test from 'node:test';
import assert from 'node:assert/strict';

import { sendTypingPresence } from './bridge_helpers.js';

const chat = '15551234567@s.whatsapp.net';

function fakeSocket() {
  const sent = [];
  return { sent, sendPresenceUpdate: async (...args) => { sent.push(args); } };
}

test('a paused state reaches WhatsApp as paused, which withdraws the indicator', async () => {
  const sock = fakeSocket();

  assert.equal(await sendTypingPresence(sock, chat, 'paused'), 'paused');
  assert.deepEqual(sock.sent, [['paused', chat]]);
});

test('no state, as an older adapter sends, still shows the indicator', async () => {
  const sock = fakeSocket();

  assert.equal(await sendTypingPresence(sock, chat, undefined), 'composing');
  assert.deepEqual(sock.sent, [['composing', chat]]);
});

test('an unknown state is never forwarded to WhatsApp as it came', async () => {
  const sock = fakeSocket();

  assert.equal(await sendTypingPresence(sock, chat, 'available'), 'composing');
  assert.deepEqual(sock.sent, [['composing', chat]]);
});
