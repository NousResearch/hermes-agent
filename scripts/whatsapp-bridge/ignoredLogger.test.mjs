import test from 'node:test';
import assert from 'node:assert/strict';

import { createIgnoredMessageLogger } from './ignoredLogger.js';

test('ignored message logger is silent unless debug logging is enabled', () => {
  const entries = [];
  const logIgnoredMessage = createIgnoredMessageLogger({
    debugEnabled: false,
    log: (entry) => entries.push(entry),
    now: () => 0,
  });

  logIgnoredMessage('allowlist_mismatch');

  assert.deepEqual(entries, []);
});

test('ignored message logger omits chat and sender identifiers', () => {
  const entries = [];
  const logIgnoredMessage = createIgnoredMessageLogger({
    debugEnabled: true,
    log: (entry) => entries.push(entry),
    now: () => 0,
  });

  logIgnoredMessage('allowlist_mismatch');

  assert.equal(entries.length, 1);
  assert.deepEqual(JSON.parse(entries[0]), {
    event: 'ignored',
    reason: 'allowlist_mismatch',
  });
  assert.equal(entries[0].includes('chatId'), false);
  assert.equal(entries[0].includes('senderId'), false);
  assert.equal(entries[0].includes('senderAliases'), false);
});

test('ignored message logger rate-limits repeated drops by reason', () => {
  const entries = [];
  let currentTime = 1000;
  const logIgnoredMessage = createIgnoredMessageLogger({
    debugEnabled: true,
    log: (entry) => entries.push(entry),
    now: () => currentTime,
    intervalMs: 60_000,
  });

  logIgnoredMessage('allowlist_mismatch');
  currentTime += 1000;
  logIgnoredMessage('allowlist_mismatch');
  currentTime += 60_000;
  logIgnoredMessage('allowlist_mismatch');

  assert.equal(entries.length, 2);
});
