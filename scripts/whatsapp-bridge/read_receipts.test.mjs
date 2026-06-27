import test from 'node:test';
import assert from 'node:assert/strict';

import { sendReadReceipt } from './read_receipts.js';

test('sendReadReceipt forwards inbound message keys to Baileys', async () => {
  const calls = [];
  const sock = {
    async readMessages(keys) {
      calls.push(keys);
    },
  };
  const msg = {
    key: {
      id: 'abc123',
      fromMe: false,
      remoteJid: '19175395595@s.whatsapp.net',
    },
  };

  const sent = await sendReadReceipt(sock, msg);
  assert.equal(sent, true);
  assert.deepEqual(calls, [[msg.key]]);
});

test('sendReadReceipt skips fromMe messages', async () => {
  const sock = {
    async readMessages() {
      throw new Error('should not be called');
    },
  };
  const msg = {
    key: {
      id: 'abc123',
      fromMe: true,
      remoteJid: '19175395595@s.whatsapp.net',
    },
  };

  const sent = await sendReadReceipt(sock, msg);
  assert.equal(sent, false);
});

test('sendReadReceipt swallows bridge ack errors', async () => {
  const errors = [];
  const sock = {
    async readMessages() {
      throw new Error('network down');
    },
  };
  const msg = {
    key: {
      id: 'abc123',
      fromMe: false,
      remoteJid: '19175395595@s.whatsapp.net',
    },
  };

  const sent = await sendReadReceipt(sock, msg, {
    error: (...args) => errors.push(args.join(' ')),
  });
  assert.equal(sent, false);
  assert.equal(errors.length, 1);
  assert.match(errors[0], /Failed to send read receipt/);
});
