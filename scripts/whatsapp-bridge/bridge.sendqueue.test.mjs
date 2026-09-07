/**
 * Regression tests for the WhatsApp bridge send queue (#33360).
 *
 * The bridge must serialise all sock.sendMessage() calls through a
 * promise-based queue so that concurrent HTTP /send requests never
 * produce overlapping Baileys socket writes.  Overlapping writes are
 * the confirmed root cause of cross-chat contamination.
 *
 * These tests exercise the queue itself — they do NOT require a live
 * WhatsApp socket.
 */

import { strict as assert } from 'node:assert';

// ------------------------------------------------------------------
// 1.  Unit test for the queue primitives
// ------------------------------------------------------------------

/**
 * Replicate the queue logic from bridge.js so we can test it in
 * isolation without importing the full module (which would trigger
 * Baileys / express side effects).
 */
function createSendQueue() {
  let _sendQueue = Promise.resolve();

  function enqueueSend(fn) {
    const task = _sendQueue.then(() => fn(), () => fn());
    _sendQueue = task.catch(() => {});
    return task;
  }

  return { enqueueSend };
}

// -- serial ordering -------------------------------------------------
{
  const { enqueueSend } = createSendQueue();
  const order = [];

  const a = enqueueSend(async () => {
    await new Promise(r => setTimeout(r, 30));
    order.push('a');
    return 'A';
  });
  const b = enqueueSend(async () => {
    order.push('b');
    return 'B';
  });
  const c = enqueueSend(async () => {
    await new Promise(r => setTimeout(r, 10));
    order.push('c');
    return 'C';
  });

  const results = await Promise.all([a, b, c]);
  assert.deepStrictEqual(results, ['A', 'B', 'C'], 'all tasks resolve');
  assert.deepStrictEqual(order, ['a', 'b', 'c'], 'tasks execute in FIFO order');
  console.log('  ✓ serial ordering');
}

// -- error isolation (one rejection does not stall the queue) --------
{
  const { enqueueSend } = createSendQueue();
  const order = [];

  const bad = enqueueSend(async () => {
    order.push('bad');
    throw new Error('boom');
  });
  const good = enqueueSend(async () => {
    order.push('good');
    return 'ok';
  });

  await assert.rejects(() => bad, /boom/, 'bad task rejects');
  const g = await good;
  assert.strictEqual(g, 'ok', 'good task still resolves');
  assert.deepStrictEqual(order, ['bad', 'good'], 'good runs after bad');
  console.log('  ✓ error isolation');
}

// -- timeout still fires (wrapped inside enqueueSend) ----------------
{
  const { enqueueSend } = createSendQueue();
  const timedOut = enqueueSend(async () => {
    await new Promise((_, reject) => setTimeout(() => reject(new Error('timeout')), 20));
  });
  await assert.rejects(() => timedOut, /timeout/, 'inner timeout propagates');
  console.log('  ✓ timeout propagation');
}

// -- concurrent enqueues maintain single-consumer semantics ----------
{
  const { enqueueSend } = createSendQueue();
  let concurrent = 0;
  let maxConcurrent = 0;

  async function tracked() {
    concurrent += 1;
    if (concurrent > maxConcurrent) maxConcurrent = concurrent;
    await new Promise(r => setTimeout(r, 5));
    concurrent -= 1;
  }

  await Promise.all(Array.from({ length: 20 }, () => enqueueSend(tracked)));
  assert.strictEqual(maxConcurrent, 1, 'never more than one in-flight');
  assert.strictEqual(concurrent, 0, 'all finished');
  console.log('  ✓ single-consumer concurrency');
}

console.log('\n✅ All send-queue tests passed.');
