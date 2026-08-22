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
import { createSerializedSendQueue } from './bridge_helpers.js';

// ------------------------------------------------------------------
// 1.  Unit test for the queue primitives
// ------------------------------------------------------------------

/**
 * Replicate the queue logic from bridge.js so we can test it in
 * isolation without importing the full module (which would trigger
 * Baileys / express side effects).
 */
const createSendQueue = createSerializedSendQueue;

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

// -- client timeout does not release queue occupancy -------------------
{
  const { enqueueSend } = createSendQueue();
  let releaseProvider;
  let secondProviderCalls = 0;
  const first = enqueueSend(
    () => new Promise(resolve => { releaseProvider = resolve; }),
    { timeoutMs: 10, timeoutError: () => new Error('client timed out') },
  );
  const second = enqueueSend(async () => { secondProviderCalls += 1; return 'second'; });

  await assert.rejects(first, /client timed out/);
  await new Promise(resolve => setTimeout(resolve, 20));
  assert.equal(secondProviderCalls, 0, 'timed-out provider still owns the queue');
  releaseProvider('late provider result');
  assert.equal(await second, 'second');
  assert.equal(secondProviderCalls, 1);
  console.log('  ✓ client timeout keeps queue occupied until provider settlement');
}

// -- preflight happens after dequeue, immediately before send ----------
{
  const { enqueueSend } = createSendQueue();
  let releaseFirst;
  let signalStarted;
  const started = new Promise(resolve => { signalStarted = resolve; });
  let fence = 0;
  let providerCalls = 0;
  const first = enqueueSend(() => {
    signalStarted();
    return new Promise(resolve => { releaseFirst = resolve; });
  });
  const second = enqueueSend(
    async () => { providerCalls += 1; },
    { beforeRun: () => {
      if (fence > 0) throw new Error('fenced');
    } },
  );
  await started;
  fence = 1;
  releaseFirst();
  await first;
  await assert.rejects(() => second, /fenced/);
  assert.equal(providerCalls, 0);
  console.log('  ✓ send preflight runs after dequeue and blocks provider call');
}

// -- queued requests expire before preflight/provider --------------------
{
  let currentTime = 100;
  const { enqueueSend } = createSerializedSendQueue({ now: () => currentTime });
  let releaseFirst;
  let signalFirstStarted;
  const firstStarted = new Promise(resolve => { signalFirstStarted = resolve; });
  let preflightCalls = 0;
  let providerCalls = 0;
  const first = enqueueSend(() => {
    signalFirstStarted();
    return new Promise(resolve => { releaseFirst = resolve; });
  });
  const expired = enqueueSend(
    async () => { providerCalls += 1; },
    {
      beforeRun: () => { preflightCalls += 1; },
      queueDeadlineAt: 110,
      queueError: () => Object.assign(new Error('queued request expired'), { code: 'SEND_QUEUE_EXPIRED' }),
    },
  );
  await firstStarted;
  currentTime = 111;
  releaseFirst();
  await first;
  await assert.rejects(expired, error => error?.code === 'SEND_QUEUE_EXPIRED');
  assert.equal(preflightCalls, 0);
  assert.equal(providerCalls, 0);
  console.log('  ✓ expired queued request never reaches preflight or provider');
}

// -- disconnected queued requests are abandoned before send -------------
{
  const { enqueueSend } = createSerializedSendQueue();
  let releaseFirst;
  let signalFirstStarted;
  const firstStarted = new Promise(resolve => { signalFirstStarted = resolve; });
  let disconnected = false;
  let providerCalls = 0;
  const first = enqueueSend(() => {
    signalFirstStarted();
    return new Promise(resolve => { releaseFirst = resolve; });
  });
  const abandoned = enqueueSend(
    async () => { providerCalls += 1; },
    { isAbandoned: () => disconnected },
  );
  await firstStarted;
  disconnected = true;
  releaseFirst();
  await first;
  await assert.rejects(abandoned, error => error?.code === 'SEND_QUEUE_ABANDONED');
  assert.equal(providerCalls, 0);
  console.log('  ✓ disconnected queued request never reaches provider');
}

console.log('\n✅ All send-queue tests passed.');
