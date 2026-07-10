import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, readFileSync, writeFileSync } from 'fs';
import { tmpdir } from 'os';
import path from 'path';

import { createIngressReceipts, receiptKey } from './ingress_receipts.js';

function tempFile() {
  return path.join(mkdtempSync(path.join(tmpdir(), 'ingress-receipts-')), 'receipts.log');
}

test('first sight records, second sight is a duplicate', () => {
  const receipts = createIngressReceipts({ filePath: tempFile() });
  assert.equal(receipts.record('123@s.whatsapp.net', 'ABC1'), true);
  assert.equal(receipts.record('123@s.whatsapp.net', 'ABC1'), false);
  assert.equal(receipts.has('123@s.whatsapp.net', 'ABC1'), true);
});

test('same message id in different chats is not a duplicate', () => {
  const receipts = createIngressReceipts({ filePath: tempFile() });
  assert.equal(receipts.record('123@s.whatsapp.net', 'ABC1'), true);
  assert.equal(receipts.record('456@g.us', 'ABC1'), true);
});

test('incident regression: 100 replays of one message process exactly once', () => {
  const receipts = createIngressReceipts({ filePath: tempFile() });
  let processed = 0;
  for (let i = 0; i < 100; i += 1) {
    if (receipts.record('40750000000@s.whatsapp.net', '3EB0REPLAYED')) processed += 1;
  }
  assert.equal(processed, 1);
});

test('receipts survive a bridge restart (new instance, same file)', () => {
  const filePath = tempFile();
  createIngressReceipts({ filePath }).record('123@s.whatsapp.net', 'ABC1');
  const reloaded = createIngressReceipts({ filePath });
  assert.equal(reloaded.has('123@s.whatsapp.net', 'ABC1'), true);
  assert.equal(reloaded.record('123@s.whatsapp.net', 'ABC1'), false);
});

test('receipt is durable before the in-memory add (crash-safety ordering)', () => {
  const filePath = tempFile();
  createIngressReceipts({ filePath }).record('123@s.whatsapp.net', 'ABC1');
  assert.ok(readFileSync(filePath, 'utf8').includes(receiptKey('123@s.whatsapp.net', 'ABC1')));
});

test('compaction keeps the newest keys and drops the oldest', () => {
  const filePath = tempFile();
  const receipts = createIngressReceipts({ filePath, maxEntries: 10, compactTo: 5 });
  for (let i = 0; i < 11; i += 1) receipts.record('c@s.whatsapp.net', `id-${i}`);
  assert.ok(receipts.size() <= 6);
  assert.equal(receipts.has('c@s.whatsapp.net', 'id-10'), true);
  assert.equal(receipts.has('c@s.whatsapp.net', 'id-0'), false);
  const reloaded = createIngressReceipts({ filePath, maxEntries: 10, compactTo: 5 });
  assert.equal(reloaded.has('c@s.whatsapp.net', 'id-10'), true);
  assert.equal(reloaded.has('c@s.whatsapp.net', 'id-0'), false);
});

test('a corrupt receipts file starts empty instead of crashing', () => {
  const filePath = tempFile();
  writeFileSync(filePath, 'not-a-log');
  const receipts = createIngressReceipts({ filePath, logger: { warn: () => {} } });
  assert.equal(receipts.record('123@s.whatsapp.net', 'ABC1'), true);
});
