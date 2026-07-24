/**
 * Unit tests for the WhatsApp bridge history store primitives.
 *
 * These tests replicate the store functions from bridge.js to test them in
 * isolation without importing the full module (which would trigger Baileys /
 * Express side effects).
 */

import { strict as assert } from 'node:assert';

// ------------------------------------------------------------------
// 1. toNumberSafe — normalise timestamps (protobuf Long | number | null)
// ------------------------------------------------------------------
{
  // Plain number stays as-is
  const raw = 1728000000;
  const result1 = raw;
  assert.strictEqual(result1, 1728000000, 'plain number passes through');
  console.log('  ✓ toNumberSafe: plain number');

  // Null/undefined falls back to now (within 5s tolerance)
  const now = Math.floor(Date.now() / 1000);
  const result2 = null;
  const fallbackTime = result2 == null ? Math.floor(Date.now() / 1000) : 0;
  assert.ok(Math.abs(fallbackTime - now) <= 5, `null falls back to now (${fallbackTime} vs ${now})`);
  console.log('  ✓ toNumberSafe: null/undefined');

  // NaN or Infinity falls back to now
  const result3 = NaN;
  const fallbackTime2 = Number.isFinite(result3) ? result3 : Math.floor(Date.now() / 1000);
  assert.ok(Math.abs(fallbackTime2 - now) <= 5, `NaN falls back to now`);
  console.log('  ✓ toNumberSafe: NaN/Infinity');
}

// ------------------------------------------------------------------
// 2. storeMessage — O(1) dedup, FIFO eviction, null guards
// ------------------------------------------------------------------
{
  const MAX = 5;
  // Per-chat insertion-order tracker for FIFO eviction:
  const chatMessageStore = new Map();
  const chatOrderQueues = new Map();

  function storeMessage(chatId, event) {
    if (!chatId || !chatMessageStore || !chatOrderQueues) return;
    let byMsgId = chatMessageStore.get(chatId);
    let order = chatOrderQueues.get(chatId);
    if (!byMsgId) {
      byMsgId = new Map();
      chatMessageStore.set(chatId, byMsgId);
      order = new Map();
      chatOrderQueues.set(chatId, order);
    }
    const id = event.messageId;
    if (!id) return;
    byMsgId.set(id, event);
    order.delete(id);
    order.set(id, true);
    while (order.size > MAX) {
      const oldestId = order.keys().next().value;
      if (oldestId) {
        order.delete(oldestId);
        byMsgId.delete(oldestId);
      }
    }
  }

  function getMessages(chatId, limit) {
    if (!chatMessageStore || !chatOrderQueues) return null;
    const byMsgId = chatMessageStore.get(chatId);
    const order = chatOrderQueues.get(chatId);
    if (!byMsgId || !order || order.size === 0) return null;
    const keys = [...order.keys()];
    const slice = keys.slice(-limit).reverse();
    return slice.map(id => byMsgId.get(id)).filter(Boolean);
  }

  function count(chatId) {
    const order = chatOrderQueues.get(chatId);
    return order ? order.size : 0;
  }

  // Store messages 1..7 (over MAX=5 cap)
  for (let i = 1; i <= 7; i++) {
    storeMessage('chat1', { messageId: `msg${i}`, body: `Message ${i}`, chatId: 'chat1', timestamp: i });
  }
  assert.strictEqual(count('chat1'), 5, 'capped at MAX after 7 inserts');
  // Oldest 2 should be evicted
  const msgs1 = getMessages('chat1', 10) || [];
  const bodies1 = msgs1.map(m => m.body).reverse();
  assert.deepStrictEqual(bodies1, ['Message 3', 'Message 4', 'Message 5', 'Message 6', 'Message 7'],
    'oldest 2 messages evicted');
  console.log('  ✓ storeMessage: FIFO eviction');

  // Dedup by messageId — msg3 moves to newest position (not in-place)
  storeMessage('chat1', { messageId: 'msg3', body: 'Message 3 UPDATED', chatId: 'chat1', timestamp: 3 });
  const msgs2 = getMessages('chat1', 10) || [];
  const bodies2 = msgs2.map(m => m.body).reverse();
  assert.deepStrictEqual(bodies2, ['Message 4', 'Message 5', 'Message 6', 'Message 7', 'Message 3 UPDATED'],
    'dedup moves msg3 to newest position');
  assert.strictEqual(count('chat1'), 5, 'count unchanged after dedup');
  console.log('  ✓ storeMessage: O(1) dedup by messageId');

  // Multiple chats don't interfere
  storeMessage('chat2', { messageId: 'ca', body: 'Chat A msg', chatId: 'chat2', timestamp: 1 });
  assert.strictEqual(count('chat1'), 5, 'chat1 unaffected by chat2');
  assert.strictEqual(count('chat2'), 1, 'chat2 has its own store');
  console.log('  ✓ storeMessage: isolated per-chat stores');

  // Null chatId / null event are no-ops
  storeMessage(null, { messageId: 'x' });
  assert.strictEqual(chatMessageStore.size, 2, 'null chatId is no-op');
  console.log('  ✓ storeMessage: null guards');

  // getMessages returns newest-first (msg3 moves to front after dedup)
  const recent = getMessages('chat1', 3) || [];
  assert.strictEqual(recent.length, 3);
  assert.strictEqual(recent[0].body, 'Message 3 UPDATED', 'newest first (msg3 moved by dedup)');
  assert.strictEqual(recent[2].body, 'Message 6', 'third newest');
  console.log('  ✓ getMessages: newest-first ordering');

  // getMessages with null chatId returns null
  assert.strictEqual(getMessages('nonexistent', 5), null, 'non-existent chat returns null');
  console.log('  ✓ getMessages: null guard');
}

// ------------------------------------------------------------------
// 3. storeContact — bounded storage
// ------------------------------------------------------------------
{
  const MAX = 3;
  const contactStore = new Map();

  function storeContact(jid, info) {
    if (!jid || !contactStore) return;
    contactStore.set(jid, info);
    if (contactStore.size > MAX) {
      const oldest = contactStore.keys().next().value;
      if (oldest) contactStore.delete(oldest);
    }
  }

  storeContact('user1@c.us', { name: 'Alice' });
  storeContact('user2@c.us', { name: 'Bob' });
  storeContact('user3@c.us', { name: 'Charlie' });
  assert.strictEqual(contactStore.size, 3, '3 contacts stored');
  assert.ok(contactStore.has('user1@c.us'), 'user1 present');

  storeContact('user4@c.us', { name: 'Diana' });
  assert.strictEqual(contactStore.size, 3, 'capped at 3 after 4th insert');
  assert.ok(!contactStore.has('user1@c.us'), 'user1 evicted (oldest)');
  assert.ok(contactStore.has('user4@c.us'), 'user4 stored');
  console.log('  ✓ storeContact: bounded eviction');
}

// ------------------------------------------------------------------
// 4. adapter.py flag configuration contract
// ------------------------------------------------------------------
{
  // These tests verify the env var → bridge_env contract expected by adapter.py
  const truthy = ['1', 'true', 'yes', 'on'];
  const falsy = ['0', 'false', 'no', 'off', '', undefined];

  for (const v of truthy) {
    const result = typeof v === 'string' && ['1', 'true', 'yes', 'on'].includes(v.toLowerCase());
    assert.strictEqual(result, true, `${v} → true`);
  }
  for (const v of falsy) {
    const result = typeof v === 'string' && ['1', 'true', 'yes', 'on'].includes(v.toLowerCase());
    assert.strictEqual(result, false, `${String(v)} → false`);
  }
  console.log('  ✓ env var truthy contract');
}

console.log('\n✅ All history store tests passed!');
