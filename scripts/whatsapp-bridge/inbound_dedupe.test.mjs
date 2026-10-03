import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';

import { createInboundDedupe } from './inbound_dedupe.js';

function tempFile() {
  return path.join(mkdtempSync(path.join(tmpdir(), 'inbound-dedupe-')), 'dedupe.json');
}

test('first sight of an id is not a duplicate; second sight is', () => {
  const dedupe = createInboundDedupe({ file: tempFile() });
  assert.equal(dedupe.checkAndRemember('msg-1'), false);
  assert.equal(dedupe.checkAndRemember('msg-1'), true);
  assert.equal(dedupe.checkAndRemember('msg-2'), false);
  assert.equal(dedupe.size(), 2);
});

test('persists ids and a fresh instance recognises them again', () => {
  const file = tempFile();
  const first = createInboundDedupe({ file });
  first.checkAndRemember('3ADBC9E571524161153A');
  first.checkAndRemember('AC3272EA9C4DFE0C81D05EE2167BD873');
  first.flush();

  const second = createInboundDedupe({ file });
  assert.equal(second.checkAndRemember('3ADBC9E571524161153A'), true);
  assert.equal(second.checkAndRemember('AC3272EA9C4DFE0C81D05EE2167BD873'), true);
  assert.equal(second.checkAndRemember('brand-new-id'), false);
});

test('tolerates a missing file', () => {
  const file = path.join(mkdtempSync(path.join(tmpdir(), 'inbound-dedupe-')), 'absent.json');
  const dedupe = createInboundDedupe({ file });
  assert.equal(dedupe.checkAndRemember('x'), false);
  assert.equal(dedupe.size(), 1);
});

test('tolerates a corrupt file and recovers on the next flush', () => {
  const file = tempFile();
  writeFileSync(file, '{not valid json!!');
  const dedupe = createInboundDedupe({ file });
  assert.equal(dedupe.checkAndRemember('y'), false);
  dedupe.flush();
  assert.deepEqual(JSON.parse(readFileSync(file, 'utf8')), ['y']);
});

test('caps the remembered set (FIFO eviction, bounded file)', () => {
  const dedupe = createInboundDedupe({ file: tempFile(), maxSize: 3 });
  for (const id of ['a', 'b', 'c', 'd']) dedupe.checkAndRemember(id);
  assert.equal(dedupe.size(), 3);
  // 'a' was evicted, so a redelivery of it is treated as new again (and its
  // re-remembering pushes the oldest survivor out — plain FIFO semantics).
  assert.equal(dedupe.checkAndRemember('a'), false);
  assert.equal(dedupe.checkAndRemember('c'), true);
});

test('rejects creation without a file path', () => {
  assert.throws(() => createInboundDedupe(), RangeError);
  assert.throws(() => createInboundDedupe({ maxSize: 8 }), RangeError);
});