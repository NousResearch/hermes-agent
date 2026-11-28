import test from 'node:test';
import assert from 'node:assert/strict';

import {
  createBoundedMessageStore,
  resolveGetMessagePayload,
} from './bridge_helpers.js';

test('missing key returns undefined (never empty conversation)', () => {
  const store = createBoundedMessageStore(8);
  const payload = resolveGetMessagePayload(store, { id: 'nope' });
  assert.equal(payload, undefined);
});

test('null/absent store returns undefined', () => {
  assert.equal(resolveGetMessagePayload(null, { id: 'x' }), undefined);
  assert.equal(resolveGetMessagePayload(undefined, { id: 'x' }), undefined);
  assert.equal(resolveGetMessagePayload(createBoundedMessageStore(), null), undefined);
  assert.equal(resolveGetMessagePayload(createBoundedMessageStore(), {}), undefined);
});

test('serves stored message content for retry', () => {
  const store = createBoundedMessageStore(8);
  store.remember({
    key: { id: 'm1' },
    message: { conversation: 'hello world' },
  });
  const payload = resolveGetMessagePayload(store, { id: 'm1' });
  assert.deepEqual(payload, { conversation: 'hello world' });
});

test('does not invent empty conversation from empty stored message', () => {
  const store = createBoundedMessageStore(8);
  store.remember({
    key: { id: 'm2' },
    message: { conversation: '' },
  });
  // Real stored empty text is technically present; still return stored object
  // (not undefined) — the bug was fabricating empty when key was missing.
  const payload = resolveGetMessagePayload(store, { id: 'm2' });
  assert.deepEqual(payload, { conversation: '' });
});

test('retry storm for unknown keys never returns empty conversation', () => {
  const store = createBoundedMessageStore(8);
  for (let i = 0; i < 1000; i += 1) {
    const payload = resolveGetMessagePayload(store, { id: `retry-${i}` });
    assert.equal(payload, undefined);
    assert.notDeepEqual(payload, { conversation: '' });
  }
});
