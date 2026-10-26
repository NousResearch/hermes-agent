import test from 'node:test';
import assert from 'node:assert/strict';

import { resolveTypingPresence } from './typing_presence.js';

test('typing presence defaults to composing for legacy requests', () => {
  assert.equal(resolveTypingPresence({ chatId: '15551234567@s.whatsapp.net' }), 'composing');
  assert.equal(resolveTypingPresence(), 'composing');
});

test('typing presence accepts composing and paused state/action values', () => {
  assert.equal(resolveTypingPresence({ state: 'composing' }), 'composing');
  assert.equal(resolveTypingPresence({ state: 'paused' }), 'paused');
  assert.equal(resolveTypingPresence({ action: 'composing' }), 'composing');
  assert.equal(resolveTypingPresence({ action: 'paused' }), 'paused');
});

test('typing presence rejects invalid or conflicting values', () => {
  assert.throws(
    () => resolveTypingPresence({ state: 'recording' }),
    /composing or paused/,
  );
  assert.throws(
    () => resolveTypingPresence({ state: 'paused', action: 'composing' }),
    /must match/,
  );
});