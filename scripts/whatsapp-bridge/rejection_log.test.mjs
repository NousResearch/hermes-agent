import test from 'node:test';
import assert from 'node:assert/strict';

import { createRejectionLog } from './rejection_log.js';

test('record() stamps every ignored event with a numeric ts', () => {
  const log = createRejectionLog();
  const before = Date.now();
  const event = log.record('allowlist_mismatch', { chatId: 'a@s.whatsapp.net', senderId: 'b@s.whatsapp.net' });
  const after = Date.now();

  assert.equal(event.event, 'ignored');
  assert.equal(event.reason, 'allowlist_mismatch');
  assert.equal(event.chatId, 'a@s.whatsapp.net');
  assert.equal(event.senderId, 'b@s.whatsapp.net');
  assert.equal(typeof event.ts, 'number');
  assert.ok(event.ts >= before && event.ts <= after);
});

test('snapshot() aggregates counts per reason across many rejections', () => {
  const log = createRejectionLog();
  for (let i = 0; i < 4177; i++) log.record('allowlist_mismatch', { chatId: 'x', senderId: 'y' });
  for (let i = 0; i < 3; i++) log.record('self_chat_mode_rejects_non_self', { chatId: 'x', senderId: 'y' });

  assert.deepEqual(log.snapshot(), {
    allowlist_mismatch: 4177,
    self_chat_mode_rejects_non_self: 3,
  });
});

test('snapshot() starts empty and is independent per log instance', () => {
  const a = createRejectionLog();
  const b = createRejectionLog();
  a.record('allowlist_mismatch', {});

  assert.deepEqual(b.snapshot(), {});
  assert.deepEqual(a.snapshot(), { allowlist_mismatch: 1 });
});
