import { test } from 'node:test';
import assert from 'node:assert/strict';

import { createBoundedMessageStore } from './bridge_helpers.js';

const msg = (id, remoteJid) => ({ key: { id, remoteJid } });

test('latestForChat returns the most recent remembered message for a chat', () => {
  const store = createBoundedMessageStore(8);
  store.remember(msg('a1', 'chatA@s.whatsapp.net'));
  store.remember(msg('b1', 'chatB@s.whatsapp.net'));
  store.remember(msg('a2', 'chatA@s.whatsapp.net'));

  assert.equal(store.latestForChat('chatA@s.whatsapp.net').key.id, 'a2');
  assert.equal(store.latestForChat('chatB@s.whatsapp.net').key.id, 'b1');
});

test('latestForChat returns null for an unknown chat or falsy id', () => {
  const store = createBoundedMessageStore(8);
  store.remember(msg('a1', 'chatA@s.whatsapp.net'));

  assert.equal(store.latestForChat('nobody@s.whatsapp.net'), null);
  assert.equal(store.latestForChat(''), null);
  assert.equal(store.latestForChat(undefined), null);
});

test('latestForChat follows re-remembered ordering', () => {
  const store = createBoundedMessageStore(8);
  store.remember(msg('a1', 'chatA@s.whatsapp.net'));
  store.remember(msg('a2', 'chatA@s.whatsapp.net'));
  // Re-remembering a1 moves it to the newest position (Map delete+set).
  store.remember(msg('a1', 'chatA@s.whatsapp.net'));

  assert.equal(store.latestForChat('chatA@s.whatsapp.net').key.id, 'a1');
});
