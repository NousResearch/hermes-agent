import assert from 'node:assert/strict';
import { test } from 'node:test';

import {
  reactionMessageStore,
  rememberMessage,
  resolveStoredMessage,
} from './bridge.js';

const chatId = '5215550000013@s.whatsapp.net';
const groupId = '120363000000000000@g.us';

function textMessage(id, remoteJid = chatId, extra = {}) {
  return {
    key: { id, remoteJid, fromMe: false, ...extra },
    message: { conversation: 'hola' },
    messageTimestamp: 1710000000,
  };
}

test('message store remembers and resolves a message by chat + id', () => {
  reactionMessageStore.clear();
  const msg = textMessage('orig-1');

  rememberMessage(msg);

  assert.equal(resolveStoredMessage(chatId, 'orig-1'), msg);
});

test('resolveStoredMessage returns undefined for an unknown message', () => {
  reactionMessageStore.clear();

  assert.equal(resolveStoredMessage(chatId, 'missing'), undefined);
});

test('stored group message retains participant on its key (needed for reactions)', () => {
  reactionMessageStore.clear();
  const msg = textMessage('grp-1', groupId, { participant: '5215550000023@s.whatsapp.net' });

  rememberMessage(msg);
  const resolved = resolveStoredMessage(groupId, 'grp-1');

  assert.equal(resolved.key.participant, '5215550000023@s.whatsapp.net');
});

test('expired entries are evicted on resolve (TTL)', () => {
  reactionMessageStore.clear();
  const msg = textMessage('old-1');
  // Remember with a seenAt far in the past so the TTL window is exceeded.
  rememberMessage(msg, Date.now() - (48 * 60 * 60 * 1000));

  assert.equal(resolveStoredMessage(chatId, 'old-1'), undefined);
});

test('rememberMessage ignores messages without a usable key', () => {
  reactionMessageStore.clear();

  rememberMessage({ message: { conversation: 'no key' } });
  rememberMessage({ key: { id: '', remoteJid: chatId } });

  assert.equal(reactionMessageStore.size, 0);
});
