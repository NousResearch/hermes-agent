import assert from 'node:assert/strict';
import { test } from 'node:test';

import { buildReactionSend, createBoundedMessageStore } from './bridge_helpers.js';

// POST /react resolves its target from the bounded message store by id: Baileys needs the
// ORIGINAL message key (including a group's `participant`) to attach or retract a reaction.

const chatId = '15551234567@s.whatsapp.net';
const groupId = '120363001234567890@g.us';

function textMessage(id, remoteJid = chatId, extra = {}) {
  return {
    key: { id, remoteJid, fromMe: false, ...extra },
    message: { conversation: 'hola' },
    messageTimestamp: 1710000000,
  };
}

function storeWith(...messages) {
  const store = createBoundedMessageStore(8);
  for (const msg of messages) store.remember(msg);
  return store;
}

test('reaction targets the stored message key', () => {
  const msg = textMessage('orig-1');
  const result = buildReactionSend({ chatId, messageId: 'orig-1', emoji: '👍', messageStore: storeWith(msg) });
  assert.deepEqual(result, { content: { react: { text: '👍', key: msg.key } } });
});

test('group reaction keeps the participant on the key', () => {
  const msg = textMessage('grp-1', groupId, { participant: '15550001111@s.whatsapp.net' });
  const { content } = buildReactionSend({
    chatId: groupId, messageId: 'grp-1', emoji: '🔥', messageStore: storeWith(msg),
  });
  assert.equal(content.react.key.participant, '15550001111@s.whatsapp.net');
  assert.equal(content.react.key.remoteJid, groupId);
});

test('an empty emoji retracts instead of being refused', () => {
  const { content } = buildReactionSend({
    chatId, messageId: 'orig-1', emoji: '', messageStore: storeWith(textMessage('orig-1')),
  });
  assert.equal(content.react.text, '');
});

test('an uncached or evicted message id is a 404, never a guessed key', () => {
  const store = createBoundedMessageStore(1);
  store.remember(textMessage('old'));
  store.remember(textMessage('new'));
  assert.deepEqual(buildReactionSend({ chatId, messageId: 'old', emoji: '👍', messageStore: store }),
    { status: 404, error: 'Referenced message not found in bridge cache' });
});

test('missing fields are a 400', () => {
  const messageStore = storeWith(textMessage('orig-1'));
  for (const body of [
    { messageId: 'orig-1', emoji: '👍' },
    { chatId, emoji: '👍' },
    { chatId, messageId: 'orig-1' },
    { chatId, messageId: 'orig-1', emoji: null },
  ]) {
    assert.equal(buildReactionSend({ ...body, messageStore }).status, 400, JSON.stringify(body));
  }
});
