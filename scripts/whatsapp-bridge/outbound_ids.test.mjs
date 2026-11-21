import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';

import { createOutboundIdTracker, withRegisteredMessageId } from './outbound_ids.js';
import { classifyOwnerMessageGate } from './owner_message_gate.js';

function tempStore(t) {
  const dir = mkdtempSync(path.join(tmpdir(), 'hermes-outbound-ids-'));
  t.after(() => rmSync(dir, { recursive: true, force: true }));
  return path.join(dir, 'outbound_ids.json');
}

function ownerGate(tracker, messageId) {
  return classifyOwnerMessageGate({
    fromMe: true,
    fromOwnerEnabled: true,
    recentlySent: tracker,
    allowlistMatches: () => true,
    messageId,
    chatId: '6281234567890@s.whatsapp.net',
  }).action;
}

test('remembers and recognises an outbound id', () => {
  const tracker = createOutboundIdTracker();
  tracker.remember('msg-1');
  assert.equal(tracker.has('msg-1'), true);
  assert.equal(tracker.has('msg-2'), false);
});

test('ignores empty / falsy ids', () => {
  const tracker = createOutboundIdTracker();
  tracker.remember(undefined);
  tracker.remember('');
  tracker.remember(null);
  assert.equal(tracker.size(), 0);
  assert.equal(tracker.has(''), false);
  assert.equal(tracker.has(undefined), false);
});

test('evicts oldest entry once the cap is exceeded', () => {
  const tracker = createOutboundIdTracker(3);
  tracker.remember('a');
  tracker.remember('b');
  tracker.remember('c');
  tracker.remember('d'); // cap=3 → 'a' should be evicted
  assert.equal(tracker.has('a'), false);
  assert.equal(tracker.has('b'), true);
  assert.equal(tracker.has('c'), true);
  assert.equal(tracker.has('d'), true);
  assert.equal(tracker.size(), 3);
});

test('rejects non-positive maxSize', () => {
  assert.throws(() => createOutboundIdTracker(0), RangeError);
  assert.throws(() => createOutboundIdTracker(-1), RangeError);
  assert.throws(() => createOutboundIdTracker(1.5), RangeError);
});

test('persisted ids survive a bridge restart', t => {
  const persistPath = tempStore(t);
  const before = createOutboundIdTracker(512, { persistPath });
  before.remember('3EB0BOT1');
  const after = createOutboundIdTracker(512, { persistPath });
  assert.equal(ownerGate(after, '3EB0BOT1'), 'drop_echo');
  assert.equal(ownerGate(after, '3EB0OWNERTYPED'), 'forward_owner');
});

test('the persisted store stays within the cap', t => {
  const persistPath = tempStore(t);
  const tracker = createOutboundIdTracker(3, { persistPath });
  for (const id of ['a', 'b', 'c', 'd']) tracker.remember(id);
  assert.deepEqual(JSON.parse(readFileSync(persistPath, 'utf8')), ['b', 'c', 'd']);
  const reloaded = createOutboundIdTracker(2, { persistPath });
  assert.equal(reloaded.has('b'), false);
  assert.equal(reloaded.has('d'), true);
  assert.equal(reloaded.size(), 2);
});

test('a missing or corrupt store starts empty instead of throwing', t => {
  const persistPath = tempStore(t);
  assert.equal(createOutboundIdTracker(8, { persistPath }).size(), 0);
  writeFileSync(persistPath, '["3EB0BOT1", "3EB0BO');
  const tracker = createOutboundIdTracker(8, { persistPath });
  assert.equal(tracker.size(), 0);
  tracker.remember('3EB0BOT2');
  assert.equal(createOutboundIdTracker(8, { persistPath }).has('3EB0BOT2'), true);
});

test('a send id is registered before the SDK call, so an early echo is not the owner', async () => {
  const tracker = createOutboundIdTracker();
  const echoes = [];
  const sock = {
    // Baileys may emit the fromMe upsert before sendMessage() settles, and a
    // timed-out send can still be delivered.
    sendMessage: async (jid, payload, options) => {
      echoes.push(ownerGate(tracker, options.messageId));
      throw new Error('sendMessage timed out after 60s');
    },
  };
  const options = withRegisteredMessageId(tracker, {}, () => '3EB0GENERATED');
  await assert.rejects(sock.sendMessage('6281234567890@s.whatsapp.net', { text: 'hi' }, options));
  assert.deepEqual(echoes, ['drop_echo']);
  assert.equal(tracker.has('3EB0GENERATED'), true);
});

test('a caller-supplied message id is kept and registered', () => {
  const tracker = createOutboundIdTracker();
  const options = withRegisteredMessageId(tracker, { quoted: null, messageId: '3EB0CALLER' }, () => 'unused');
  assert.deepEqual(options, { quoted: null, messageId: '3EB0CALLER' });
  assert.equal(tracker.has('3EB0CALLER'), true);
});
