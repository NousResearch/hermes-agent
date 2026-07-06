import assert from 'node:assert/strict';
import test from 'node:test';

import { buildReactionPayload } from './reaction.js';

test('buildReactionPayload targets a direct chat message', () => {
  const payload = buildReactionPayload({
    chatId: '123@s.whatsapp.net',
    messageId: 'msg-1',
    emoji: '👀',
    fromMe: false,
  });

  assert.deepEqual(payload, {
    react: {
      text: '👀',
      key: {
        remoteJid: '123@s.whatsapp.net',
        id: 'msg-1',
        fromMe: false,
      },
    },
  });
});

test('buildReactionPayload includes group participant when provided', () => {
  const payload = buildReactionPayload({
    chatId: '123@g.us',
    messageId: 'msg-2',
    emoji: '✅',
    senderId: '456@s.whatsapp.net',
    fromMe: false,
  });

  assert.equal(payload.react.key.participant, '456@s.whatsapp.net');
});

test('buildReactionPayload preserves fromMe for self-chat messages', () => {
  const payload = buildReactionPayload({
    chatId: '123@s.whatsapp.net',
    messageId: 'msg-3',
    emoji: '❌',
    fromMe: true,
  });

  assert.equal(payload.react.key.fromMe, true);
});
