/**
 * Unit test for the WhatsApp bridge send serialization fix (#33360)
 *
 * Simulates concurrent sock.sendMessage() calls to verify that:
 * 1. Sends are serialized (only one active at a time)
 * 2. Each send still targets the correct chatId
 * 3. No cross-chat contamination occurs under load
 *
 * Usage:
 *   cd scripts/whatsapp-bridge
 *   node bridge.sendqueue.test.mjs
 */

import { strict as assert } from 'assert';

// ── Replicate the serialization logic from bridge.js ────────────────
const sendQueue = [];
let sendActive = false;

async function _drainSendQueue() {
  if (sendActive) return;
  sendActive = true;
  while (sendQueue.length > 0) {
    const { chatId, payload, resolve, reject } = sendQueue.shift();
    try {
      const result = await _sendWithTimeoutUnlocked(chatId, payload);
      resolve(result);
    } catch (err) {
      reject(err);
    }
  }
  sendActive = false;
}

function _enqueueSend(chatId, payload) {
  return new Promise((resolve, reject) => {
    sendQueue.push({ chatId, payload, resolve, reject });
    _drainSendQueue();
  });
}

// Mock sock that tracks concurrent calls and simulates slow sends
let activeSends = 0;
let maxConcurrentSends = 0;
const sendLog = [];

async function _sendWithTimeoutUnlocked(chatId, payload) {
  activeSends++;
  if (activeSends > maxConcurrentSends) {
    maxConcurrentSends = activeSends;
  }
  // Simulate a slow send (100ms)
  await new Promise(r => setTimeout(r, 100));
  sendLog.push({ chatId, text: payload.text });
  activeSends--;
  return { key: { id: `msg-${Date.now()}-${Math.random().toString(36).slice(2, 7)}` } };
}

function sendWithTimeout(chatId, payload) {
  return _enqueueSend(chatId, payload);
}

// ── Tests ─────────────────────────────────────────────────────────────

async function testSerialization() {
  console.log('Test 1: Concurrent sends to different chats are serialized');
  // Reset state
  sendLog.length = 0;
  maxConcurrentSends = 0;
  activeSends = 0;
  const chatA = '120363406842679669@g.us';
  const chatB = '81072524161065@lid';

  // Fire 4 sends concurrently: 2 to chat A, 2 to chat B
  const promises = [
    sendWithTimeout(chatA, { text: 'msg-A1' }),
    sendWithTimeout(chatB, { text: 'msg-B1' }),
    sendWithTimeout(chatA, { text: 'msg-A2' }),
    sendWithTimeout(chatB, { text: 'msg-B2' }),
  ];

  await Promise.all(promises);

  // Assert no concurrent sends happened
  assert.strictEqual(maxConcurrentSends, 1, `Expected max 1 concurrent send, got ${maxConcurrentSends}`);

  // Assert every message went to the correct chat
  const aMessages = sendLog.filter(s => s.chatId === chatA).map(s => s.text);
  const bMessages = sendLog.filter(s => s.chatId === chatB).map(s => s.text);

  assert.deepStrictEqual(aMessages, ['msg-A1', 'msg-A2']);
  assert.deepStrictEqual(bMessages, ['msg-B1', 'msg-B2']);

  console.log('  ✓ PASSED: sends are serialized, no cross-chat contamination');
}

async function testStress() {
  console.log('Test 2: Stress test with 50 concurrent sends');
  // Reset state
  sendLog.length = 0;
  maxConcurrentSends = 0;
  activeSends = 0;
  const chatIds = [
    'group1@g.us',
    'group2@g.us',
    'user1@lid',
    'user2@lid',
    'user3@s.whatsapp.net',
  ];

  const promises = [];
  const expected = [];
  for (let i = 0; i < 50; i++) {
    const chatId = chatIds[i % chatIds.length];
    const text = `stress-${i}`;
    expected.push({ chatId, text });
    promises.push(sendWithTimeout(chatId, { text }));
  }

  await Promise.all(promises);

  assert.strictEqual(maxConcurrentSends, 1, `Expected max 1 concurrent send under stress, got ${maxConcurrentSends}`);
  assert.strictEqual(sendLog.length, 50, `Expected 50 sends, got ${sendLog.length}`);

  for (let i = 0; i < 50; i++) {
    assert.strictEqual(sendLog[i].chatId, expected[i].chatId, `Send ${i} went to wrong chat`);
    assert.strictEqual(sendLog[i].text, expected[i].text, `Send ${i} had wrong text`);
  }

  console.log('  ✓ PASSED: 50 concurrent sends serialized correctly');
}

// ── Runner ──────────────────────────────────────────────────────────

async function main() {
  console.log('=== WhatsApp Bridge Send Serialization Tests ===\n');
  await testSerialization();
  await testStress();
  console.log('\nAll tests passed ✓');
}

main().catch(err => {
  console.error('Test failed:', err);
  process.exit(1);
});
