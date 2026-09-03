/**
 * Unit tests for the WhatsApp bridge history store.
 *
 * These tests import the REAL production store from history_store.js —
 * the same code bridge.js runs — so production/endpoint regressions
 * cannot slip through (previously the tests copied the functions).
 */

import { strict as assert } from 'node:assert';
import { createHistoryStore, parseTruthyEnv } from './history_store.js';

// ------------------------------------------------------------------
// 1. toNumberSafe — normalise timestamps (protobuf Long | number | null)
// ------------------------------------------------------------------
{
  const { toNumberSafe } = createHistoryStore({ enabled: true });

  // Plain number stays as-is
  assert.strictEqual(toNumberSafe(1728000000), 1728000000, 'plain number passes through');
  console.log('  ✓ toNumberSafe: plain number');

  // Null/undefined falls back to now (within 5s tolerance)
  const now = Math.floor(Date.now() / 1000);
  const fallbackTime = toNumberSafe(null);
  assert.ok(Math.abs(fallbackTime - now) <= 5, `null falls back to now (${fallbackTime} vs ${now})`);
  console.log('  ✓ toNumberSafe: null/undefined');

  // NaN or Infinity falls back to now
  const fallbackTime2 = toNumberSafe(NaN);
  assert.ok(Math.abs(fallbackTime2 - now) <= 5, 'NaN falls back to now');
  console.log('  ✓ toNumberSafe: NaN/Infinity');

  // Protobuf Long object { low, high } with toNumber() is unwrapped
  const protoLong = { low: 1728000000 >>> 0, high: 0, toNumber: () => 1728000000 };
  assert.strictEqual(toNumberSafe(protoLong), 1728000000, 'protobuf Long unwrapped via toNumber()');
  console.log('  ✓ toNumberSafe: protobuf Long object');
}

// ------------------------------------------------------------------
// 2. storeMessage — O(1) dedup, FIFO eviction, null guards
// ------------------------------------------------------------------
{
  const MAX = 5;
  const store = createHistoryStore({ enabled: true, maxMessagesPerChat: MAX, maxContacts: 10 });

  // Store messages 1..7 (over MAX=5 cap)
  for (let i = 1; i <= 7; i++) {
    store.storeMessage('chat1', { messageId: `msg${i}`, body: `Message ${i}`, chatId: 'chat1', timestamp: i });
  }
  assert.strictEqual(store.count('chat1'), 5, 'capped at MAX after 7 inserts');
  // Oldest 2 should be evicted
  const msgs1 = store.getMessages('chat1', 10) || [];
  const bodies1 = msgs1.map(m => m.body).reverse();
  assert.deepStrictEqual(bodies1, ['Message 3', 'Message 4', 'Message 5', 'Message 6', 'Message 7'],
    'oldest 2 messages evicted');
  console.log('  ✓ storeMessage: FIFO eviction');

  // Dedup by messageId — msg3 moves to newest position (not in-place)
  store.storeMessage('chat1', { messageId: 'msg3', body: 'Message 3 UPDATED', chatId: 'chat1', timestamp: 3 });
  const msgs2 = store.getMessages('chat1', 10) || [];
  const bodies2 = msgs2.map(m => m.body).reverse();
  assert.deepStrictEqual(bodies2, ['Message 4', 'Message 5', 'Message 6', 'Message 7', 'Message 3 UPDATED'],
    'dedup moves msg3 to newest position');
  assert.strictEqual(store.count('chat1'), 5, 'count unchanged after dedup');
  console.log('  ✓ storeMessage: O(1) dedup by messageId');

  // Multiple chats don't interfere
  store.storeMessage('chat2', { messageId: 'ca', body: 'Chat A msg', chatId: 'chat2', timestamp: 1 });
  assert.strictEqual(store.count('chat1'), 5, 'chat1 unaffected by chat2');
  assert.strictEqual(store.count('chat2'), 1, 'chat2 has its own store');
  console.log('  ✓ storeMessage: isolated per-chat stores');

  // Null chatId / null event are no-ops
  store.storeMessage(null, { messageId: 'x' });
  assert.strictEqual(store.chatMessageStore.size, 2, 'null chatId is no-op');
  store.storeMessage('chat1', null);
  assert.strictEqual(store.count('chat1'), 5, 'null event is no-op');
  console.log('  ✓ storeMessage: null guards');

  // getMessages returns newest-first (msg3 moves to front after dedup)
  const recent = store.getMessages('chat1', 3) || [];
  assert.strictEqual(recent.length, 3);
  assert.strictEqual(recent[0].body, 'Message 3 UPDATED', 'newest first (msg3 moved by dedup)');
  assert.strictEqual(recent[2].body, 'Message 6', 'third newest');
  console.log('  ✓ getMessages: newest-first ordering');

  // getMessages with null chatId returns null
  assert.strictEqual(store.getMessages('nonexistent', 5), null, 'non-existent chat returns null');
  console.log('  ✓ getMessages: null guard');
}

// ------------------------------------------------------------------
// 3. storeContact — bounded storage
// ------------------------------------------------------------------
{
  const MAX = 3;
  const store = createHistoryStore({ enabled: true, maxMessagesPerChat: 10, maxContacts: MAX });

  store.storeContact('user1@c.us', { name: 'Alice' });
  store.storeContact('user2@c.us', { name: 'Bob' });
  store.storeContact('user3@c.us', { name: 'Charlie' });
  assert.strictEqual(store.contactStore.size, 3, '3 contacts stored');
  assert.ok(store.contactStore.has('user1@c.us'), 'user1 present');

  store.storeContact('user4@c.us', { name: 'Diana' });
  assert.strictEqual(store.contactStore.size, 3, 'capped at 3 after 4th insert');
  assert.ok(!store.contactStore.has('user1@c.us'), 'user1 evicted (oldest)');
  assert.ok(store.contactStore.has('user4@c.us'), 'user4 stored');
  console.log('  ✓ storeContact: bounded eviction');
}

// ------------------------------------------------------------------
// 4. Disabled store — mutators are no-ops, reads return null/0
// ------------------------------------------------------------------
{
  const store = createHistoryStore({ enabled: false });
  assert.strictEqual(store.chatMessageStore, null, 'stores are null when disabled');
  assert.strictEqual(store.contactStore, null, 'contact store null when disabled');
  store.storeMessage('chat1', { messageId: 'x', body: 'y' });
  store.storeContact('user1@c.us', { name: 'A' });
  assert.strictEqual(store.count('chat1'), 0, 'count is 0 when disabled');
  assert.strictEqual(store.getMessages('chat1', 5), null, 'getMessages null when disabled');
  console.log('  ✓ disabled store: no-ops');
}

// ------------------------------------------------------------------
// 5. adapter.py flag configuration contract
// ------------------------------------------------------------------
{
  // These tests verify the truthy parse shared by bridge.js and the
  // adapter's config.yaml → bridge_env contract.
  const truthy = ['1', 'true', 'yes', 'on', 'TRUE', 'Yes'];
  const falsy = ['0', 'false', 'no', 'off', '', undefined, null, 1, true];

  for (const v of truthy) {
    assert.strictEqual(parseTruthyEnv(v), true, `${String(v)} → true`);
  }
  for (const v of falsy) {
    assert.strictEqual(parseTruthyEnv(v), false, `${String(v)} → false`);
  }
  console.log('  ✓ parseTruthyEnv contract');
}

console.log('\n✅ All history store tests passed (against production helper)');
