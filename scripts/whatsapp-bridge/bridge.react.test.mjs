/**
 * Unit tests for the WhatsApp reaction payload helper (POST /react).
 *
 * These tests avoid importing bridge.js because that file starts an HTTP
 * server and Baileys socket at module load. Keep the helper module pure.
 */

import { strict as assert } from 'node:assert';
import { test } from 'node:test';
import { generateWAMessageContent, proto } from '@whiskeysockets/baileys';

import { buildReactionPayload, createBoundedMessageStore } from './bridge_helpers.js';

test('react and unreact preserve the cached target key for own and group messages', async () => {
  const messageStore = createBoundedMessageStore();
  const keys = [
    { remoteJid: '15551234567@s.whatsapp.net', id: 'own-message', fromMe: true },
    {
      remoteJid: '123456789@g.us', id: 'group-message', fromMe: false,
      participant: '15557654321@s.whatsapp.net',
    },
    {
      remoteJid: '987654321@lid', remoteJidAlt: '15551234567@s.whatsapp.net',
      id: 'lid-message', fromMe: true,
    },
  ];
  for (const key of keys) {
    Object.freeze(key);
    messageStore.remember({ key, message: { conversation: 'original message' } });
    for (const emoji of ['👍', '']) {
      const payload = buildReactionPayload({
        chatId: key.remoteJidAlt || key.remoteJid, messageId: key.id, emoji,
        // The adapter cannot infer ownership for an explicit message_id.
        fromMe: false,
      }, { messageStore });
      assert.deepEqual(payload.react, { text: emoji, key });
      const message = await generateWAMessageContent(payload, {});
      const wire = proto.Message.decode(proto.Message.encode(message).finish()).reactionMessage;
      assert.equal(wire.text, emoji);
      assert.equal(wire.key.id, key.id);
      assert.equal(wire.key.remoteJid, key.remoteJid);
      assert.equal(wire.key.fromMe, key.fromMe);
      if (key.participant) assert.equal(wire.key.participant, key.participant);
    }
  }
  // Baileys puts the participant on the message itself for our group sends.
  messageStore.remember({
    key: { remoteJid: '123456789@g.us', id: 'own-group-message', fromMe: true },
    participant: '15551234567@s.whatsapp.net',
    message: { conversation: 'original group reply' },
  });
  const payload = buildReactionPayload({
    chatId: '123456789@g.us', messageId: 'own-group-message', emoji: '👍', fromMe: false,
  }, { messageStore });
  assert.deepEqual(payload.react.key, {
    remoteJid: '123456789@g.us', id: 'own-group-message', fromMe: true,
    participant: '15551234567@s.whatsapp.net',
  });
});

test('reaction targets reject another chat or a group without a known participant', () => {
  const messageStore = createBoundedMessageStore();
  messageStore.remember({
    key: { remoteJid: '15551234567@s.whatsapp.net', id: 'private-message', fromMe: true },
    message: { conversation: 'original message' },
  });
  assert.throws(() => buildReactionPayload({
    chatId: '15557654321@s.whatsapp.net', messageId: 'private-message', emoji: '👍',
  }, { messageStore }), /another chat/);
  assert.throws(() => buildReactionPayload({
    chatId: '123456789@g.us', messageId: 'uncached-message', emoji: '👍',
  }, { messageStore }), /participant/);
});

{
  const payload = buildReactionPayload({
    chatId: '15551234567@s.whatsapp.net',
    messageId: 'msg-1',
    emoji: '👍',
  });
  assert.deepEqual(payload, {
    react: {
      text: '👍',
      key: {
        remoteJid: '15551234567@s.whatsapp.net',
        fromMe: false,
        id: 'msg-1',
      },
    },
  });
  console.log('  ✓ builds a native Baileys reaction payload');
}

{
  // WhatsApp retracts a reaction when the react text is '' — an empty emoji
  // is the unreact path, not a validation error.
  const payload = buildReactionPayload({
    chatId: '15551234567@s.whatsapp.net',
    messageId: 'msg-1',
    emoji: '',
  });
  assert.equal(payload.react.text, '');
  assert.equal(payload.react.key.id, 'msg-1');
  console.log('  ✓ empty emoji builds an unreact payload');
}

{
  const payload = buildReactionPayload({
    chatId: '15551234567@s.whatsapp.net',
    messageId: 'msg-1',
    emoji: '👍',
    fromMe: true,
  });
  assert.equal(payload.react.key.fromMe, true);

  const coerced = buildReactionPayload({
    chatId: '15551234567@s.whatsapp.net',
    messageId: 'msg-1',
    emoji: '👍',
    fromMe: 'true',
  });
  assert.equal(coerced.react.key.fromMe, true);
  console.log('  ✓ fromMe targets our own message (accepts bool or "true")');
}

{
  assert.throws(
    () => buildReactionPayload({ messageId: 'msg-1', emoji: '👍' }),
    /chatId is required/,
  );
  assert.throws(
    () => buildReactionPayload({ chatId: 'c@s.whatsapp.net', emoji: '👍' }),
    /messageId is required/,
  );
  assert.throws(
    () => buildReactionPayload({ chatId: 'c@s.whatsapp.net', messageId: 'msg-1' }),
    /emoji is required/,
  );
  console.log('  ✓ missing chatId/messageId/emoji are rejected');
}

console.log('\n✅ All WhatsApp reaction bridge helper tests passed.');
