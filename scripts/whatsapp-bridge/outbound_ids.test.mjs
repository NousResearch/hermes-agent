import test from 'node:test';
import assert from 'node:assert/strict';
import { appendFileSync, mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';

import {
  createDurableOutboundIdTracker,
  createOutboundIdTracker,
  sendWithProvenance,
} from './outbound_ids.js';

test('durable outbound ids survive a bridge restart', () => {
  const directory = mkdtempSync(path.join(tmpdir(), 'hermes-outbound-'));
  const tracker = createDurableOutboundIdTracker({ directory });
  tracker.remember('connector-1');

  assert.equal(createDurableOutboundIdTracker({ directory }).has('connector-1'), true);
});

test('durable tracker never evicts ids that are still inside the replay window', () => {
  const directory = mkdtempSync(path.join(tmpdir(), 'hermes-outbound-window-'));
  const tracker = createDurableOutboundIdTracker({
    directory,
    maxSize: 2,
    retentionMs: 60_000,
    now: () => 1_000,
  });
  tracker.remember('connector-a');
  tracker.remember('connector-b');
  tracker.remember('connector-c');

  assert.equal(tracker.has('connector-a'), true);
  assert.equal(tracker.size(), 3);
});

test('durable tracker repairs an incomplete final journal record after crash', () => {
  const directory = mkdtempSync(path.join(tmpdir(), 'hermes-outbound-tail-'));
  const tracker = createDurableOutboundIdTracker({ directory });
  tracker.remember('connector-valid');
  appendFileSync(path.join(directory, 'connector-outbound-ids.json'), '{"id":"partial');

  const reopened = createDurableOutboundIdTracker({ directory });
  assert.equal(reopened.has('connector-valid'), true);
  reopened.remember('connector-after-repair');
  assert.equal(createDurableOutboundIdTracker({ directory }).has('connector-after-repair'), true);
});

test('send provenance is durable before the connector send starts and remains after failure', async () => {
  const directory = mkdtempSync(path.join(tmpdir(), 'hermes-outbound-race-'));
  const tracker = createDurableOutboundIdTracker({ directory });
  await assert.rejects(
    sendWithProvenance({
      tracker,
      messageId: 'reserved-1',
      send: async () => {
        assert.equal(createDurableOutboundIdTracker({ directory }).has('reserved-1'), true);
        throw new Error('simulated send failure');
      },
    }),
    /simulated send failure/,
  );
  assert.equal(createDurableOutboundIdTracker({ directory }).has('reserved-1'), true);
});

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

test('cap holds across many inserts (bounded memory)', () => {
  const tracker = createOutboundIdTracker(8);
  for (let i = 0; i < 100; i += 1) {
    tracker.remember(`id-${i}`);
  }
  assert.equal(tracker.size(), 8);
  // Oldest (id-0..id-91) should be gone, latest 8 retained.
  assert.equal(tracker.has('id-0'), false);
  assert.equal(tracker.has('id-91'), false);
  assert.equal(tracker.has('id-92'), true);
  assert.equal(tracker.has('id-99'), true);
});

test('re-remembering an existing id does not promote it (FIFO, not LRU)', () => {
  // Insertion-order semantics: re-adding doesn't move it forward in
  // Set iteration order. This is intentional — we don't need recency,
  // just bounded membership.  Pin the actual behaviour so future
  // refactors don't accidentally introduce LRU refresh semantics.
  const tracker = createOutboundIdTracker(2);
  tracker.remember('a');
  tracker.remember('b');
  tracker.remember('a'); // no-op for ordering
  tracker.remember('c'); // evicts 'a' (oldest by insertion)
  assert.equal(tracker.has('a'), false);
  assert.equal(tracker.has('b'), true);
  assert.equal(tracker.has('c'), true);
});

test('rejects non-positive maxSize', () => {
  assert.throws(() => createOutboundIdTracker(0), RangeError);
  assert.throws(() => createOutboundIdTracker(-1), RangeError);
  assert.throws(() => createOutboundIdTracker(1.5), RangeError);
});
