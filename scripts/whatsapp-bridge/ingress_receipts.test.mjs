import { test } from 'node:test';
import assert from 'node:assert/strict';
import { chmodSync, existsSync, mkdtempSync, utimesSync, writeFileSync } from 'fs';
import { execFile } from 'child_process';
import { tmpdir } from 'os';
import path from 'path';
import { promisify } from 'util';

import { createIngressReceipts, receiptKey } from './ingress_receipts.js';

const execFileAsync = promisify(execFile);

// chmod-based failure injection is a no-op for root (e.g. some CI containers).
const isRoot = process.getuid?.() === 0;

function tempDir() {
  return path.join(mkdtempSync(path.join(tmpdir(), 'ingress-receipts-')), 'receipts');
}

function claimPath(dirPath, chatId, messageId) {
  return path.join(dirPath, encodeURIComponent(receiptKey(chatId, messageId)));
}

test('first sight records, second sight is a duplicate', () => {
  const receipts = createIngressReceipts({ dirPath: tempDir() });
  assert.equal(receipts.record('123@s.whatsapp.net', 'ABC1'), true);
  assert.equal(receipts.record('123@s.whatsapp.net', 'ABC1'), false);
  assert.equal(receipts.has('123@s.whatsapp.net', 'ABC1'), true);
});

test('same message id in different chats is not a duplicate', () => {
  const receipts = createIngressReceipts({ dirPath: tempDir() });
  assert.equal(receipts.record('123@s.whatsapp.net', 'ABC1'), true);
  assert.equal(receipts.record('456@g.us', 'ABC1'), true);
});

test('incident regression: 100 replays of one message process exactly once', () => {
  const receipts = createIngressReceipts({ dirPath: tempDir() });
  let processed = 0;
  for (let i = 0; i < 100; i += 1) {
    if (receipts.record('40750000000@s.whatsapp.net', '3EB0REPLAYED')) processed += 1;
  }
  assert.equal(processed, 1);
});

test('receipts survive a bridge restart (new instance, same dir)', () => {
  const dirPath = tempDir();
  createIngressReceipts({ dirPath }).record('123@s.whatsapp.net', 'ABC1');
  const reloaded = createIngressReceipts({ dirPath });
  assert.equal(reloaded.has('123@s.whatsapp.net', 'ABC1'), true);
  assert.equal(reloaded.record('123@s.whatsapp.net', 'ABC1'), false);
});

test('the claim is durable before record() reports a win (crash-safety ordering)', () => {
  const dirPath = tempDir();
  createIngressReceipts({ dirPath }).record('123@s.whatsapp.net', 'ABC1');
  assert.ok(existsSync(claimPath(dirPath, '123@s.whatsapp.net', 'ABC1')));
});

test('incident regression: concurrent processes claim a message exactly once', async () => {
  const dirPath = tempDir();
  const workerPath = path.join(path.dirname(dirPath), 'race_worker.mjs');
  const moduleUrl = new URL('./ingress_receipts.js', import.meta.url).href;
  writeFileSync(workerPath, [
    `import { createIngressReceipts } from ${JSON.stringify(moduleUrl)};`,
    'const receipts = createIngressReceipts({ dirPath: process.argv[2] });',
    // Each worker starts with its own empty in-memory view, so only the
    // on-disk exclusive create arbitrates the race.
    "process.stdout.write(receipts.record('race@s.whatsapp.net', '3EB0RACE') ? '1' : '0');",
  ].join('\n'));
  const runs = await Promise.all(
    Array.from({ length: 8 }, () => execFileAsync(process.execPath, [workerPath, dirPath])),
  );
  const wins = runs.reduce((sum, { stdout }) => sum + Number(stdout), 0);
  assert.equal(wins, 1);
});

test('compaction keeps the newest receipts and drops the oldest', () => {
  const dirPath = tempDir();
  const first = createIngressReceipts({ dirPath });
  for (let i = 0; i < 11; i += 1) first.record('c@s.whatsapp.net', `id-${i}`);
  // Recency is judged by mtime; spread the claims out so it is well-defined.
  const base = Date.now() / 1000 - 3600;
  for (let i = 0; i < 11; i += 1) {
    utimesSync(claimPath(dirPath, 'c@s.whatsapp.net', `id-${i}`), base + i, base + i);
  }
  const compacted = createIngressReceipts({ dirPath, maxEntries: 10, compactTo: 5 });
  assert.equal(compacted.size(), 5);
  assert.equal(compacted.has('c@s.whatsapp.net', 'id-10'), true);
  assert.equal(compacted.has('c@s.whatsapp.net', 'id-0'), false);
  const reloaded = createIngressReceipts({ dirPath, maxEntries: 10, compactTo: 5 });
  assert.equal(reloaded.has('c@s.whatsapp.net', 'id-10'), true);
  assert.equal(reloaded.has('c@s.whatsapp.net', 'id-0'), false);
});

test('an unwritable receipts dir fails open and does not poison later replays', { skip: isRoot }, () => {
  const dirPath = tempDir();
  const warnings = [];
  const receipts = createIngressReceipts({ dirPath, logger: { warn: (m) => warnings.push(m) } });
  chmodSync(dirPath, 0o500);
  try {
    // No durable receipt could be written: forward, and do NOT remember the
    // key — a replay must not be dropped on the strength of a failed write.
    assert.equal(receipts.record('123@s.whatsapp.net', 'ABC1'), true);
    assert.equal(receipts.record('123@s.whatsapp.net', 'ABC1'), true);
    assert.equal(warnings.length, 2);
  } finally {
    chmodSync(dirPath, 0o700);
  }
  // Once storage recovers, deduplication resumes.
  assert.equal(receipts.record('123@s.whatsapp.net', 'ABC1'), true);
  assert.equal(receipts.record('123@s.whatsapp.net', 'ABC1'), false);
});

test('an unreadable receipts dir degrades at load instead of crashing', { skip: isRoot }, () => {
  const dirPath = tempDir();
  createIngressReceipts({ dirPath }).record('123@s.whatsapp.net', 'ABC1');
  chmodSync(dirPath, 0o000);
  try {
    const warnings = [];
    const receipts = createIngressReceipts({ dirPath, logger: { warn: (m) => warnings.push(m) } });
    assert.equal(warnings.length, 1);
    // Degraded means forwarding, never dropping.
    assert.equal(receipts.record('456@s.whatsapp.net', 'DEF2'), true);
  } finally {
    chmodSync(dirPath, 0o700);
  }
});

test('a receipts dir that cannot be created does not take the bridge down', () => {
  const blocker = path.join(mkdtempSync(path.join(tmpdir(), 'ingress-receipts-')), 'blocker');
  writeFileSync(blocker, '');
  const dirPath = path.join(blocker, 'receipts');
  const receipts = createIngressReceipts({ dirPath, logger: { warn: () => {} } });
  assert.equal(receipts.record('123@s.whatsapp.net', 'ABC1'), true);
  assert.equal(receipts.record('123@s.whatsapp.net', 'ABC1'), true);
});

test('storage failure during a compaction-eligible claim keeps forwarding', { skip: isRoot }, () => {
  const dirPath = tempDir();
  const warnings = [];
  const receipts = createIngressReceipts({
    dirPath,
    maxEntries: 3,
    compactTo: 1,
    logger: { warn: (m) => warnings.push(m) },
  });
  for (let i = 0; i < 3; i += 1) receipts.record('c@s.whatsapp.net', `id-${i}`);
  chmodSync(dirPath, 0o500);
  try {
    assert.equal(receipts.record('c@s.whatsapp.net', 'id-3'), true);
    assert.ok(warnings.length >= 1);
  } finally {
    chmodSync(dirPath, 0o700);
  }
  // Once writable again, the claim lands and compaction runs without a crash.
  assert.equal(receipts.record('c@s.whatsapp.net', 'id-3'), true);
  assert.equal(receipts.record('c@s.whatsapp.net', 'id-3'), false);
  assert.ok(receipts.size() <= 2);
});
