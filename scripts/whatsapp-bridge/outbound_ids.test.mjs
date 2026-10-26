import test from 'node:test';
import assert from 'node:assert/strict';

import { createOutboundIdTracker } from './outbound_ids.js';

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
