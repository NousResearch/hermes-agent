import assert from 'node:assert/strict';
import { test } from 'node:test';

import { createBoundedMessageStore } from './bridge_helpers.js';

const chatId = '5215550000013@s.whatsapp.net';
const groupId = '120363000000000000@g.us';

function textMessage(id, remoteJid = chatId, extra = {}) {
  return {
    key: { id, remoteJid, fromMe: false, ...extra },
    message: { conversation: 'hola' },
    messageTimestamp: 1710000000,
  };
}

// The /react endpoint resolves the reaction target from the shared bounded
// message store by id — Baileys needs the ORIGINAL message key (including the
// group `participant`) to attach a reaction, so these tests pin the store
// behavior /react depends on.

test('store remembers and resolves a message by id', () => {
  const store = createBoundedMessageStore(8);
  const msg = textMessage('orig-1');
  store.remember(msg);
  assert.equal(store.get('orig-1'), msg);
});

test('store returns null for an unknown message id', () => {
  const store = createBoundedMessageStore(8);
  assert.equal(store.get('missing'), null);
});

test('stored group message retains participant on its key (needed for reactions)', () => {
  const store = createBoundedMessageStore(8);
  const msg = textMessage('grp-1', groupId, { participant: '5215550000023@s.whatsapp.net' });
  store.remember(msg);
  const resolved = store.get('grp-1');
  assert.equal(resolved.key.participant, '5215550000023@s.whatsapp.net');
  assert.equal(resolved.key.remoteJid, groupId);
});

test('store ignores messages without a usable key', () => {
  const store = createBoundedMessageStore(8);
  store.remember({ message: { conversation: 'no key' } });
  store.remember(null);
  assert.equal(store.get(''), null);
});

test('LRU eviction drops the oldest message once over capacity', () => {
  const store = createBoundedMessageStore(2);
  store.remember(textMessage('a'));
  store.remember(textMessage('b'));
  store.remember(textMessage('c'));
  assert.equal(store.get('a'), null); // evicted
  assert.ok(store.get('b'));
  assert.ok(store.get('c'));
});
