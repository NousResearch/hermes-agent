/**
 * Unit tests for the WhatsApp reaction payload helper (POST /react).
 *
 * These tests avoid importing bridge.js because that file starts an HTTP
 * server and Baileys socket at module load. Keep the helper module pure.
 */

import { strict as assert } from 'node:assert';

import { buildReactionPayload } from './bridge_helpers.js';

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
