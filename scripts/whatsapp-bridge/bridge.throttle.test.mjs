/**
 * Unit tests for the optional pre-send throttle (WHATSAPP_SEND_DELAY_MS).
 *
 * These tests avoid importing bridge.js because that file starts an HTTP
 * server and Baileys socket at module load. The throttle is a pure helper
 * (createSendThrottle) with the send queue and the sleep injected, so a
 * recording fake stands in for the real clock.
 */

import { strict as assert } from 'node:assert';

import { createSendThrottle } from './bridge_helpers.js';

// -- legacy default: 0/absent/unparsable adds no wait -----------------------
{
  for (const [label, configured] of [
    ['undefined', undefined],
    ['empty string', ''],
    ['zero', '0'],
    ['unparsable', 'abc'],
    ['negative (clamped)', '-5'],
  ]) {
    const ops = [];
    const enqueueSend = (fn) => { ops.push('enqueue'); return fn(); };
    const throttled = createSendThrottle(configured, {
      enqueueSend,
      sleep: async (ms) => { ops.push(`sleep:${ms}`); },
    });
    const result = await throttled(async () => { ops.push('send'); return 'sent'; });
    assert.equal(result, 'sent');
    assert.deepEqual(ops, ['enqueue', 'send']);
    console.log(`  ✓ ${label} delay keeps the legacy unthrottled queue`);
  }
}

// -- configured delay: the wait happens inside the queued task --------------
{
  const ops = [];
  const sleeps = [];
  const throttled = createSendThrottle('250', {
    enqueueSend: (fn) => fn(),
    sleep: async (ms) => { sleeps.push(ms); ops.push('sleep:start'); ops.push('sleep:end'); },
  });
  const result = await throttled(async () => { ops.push('send'); return 'sent'; });
  assert.equal(result, 'sent');
  assert.deepEqual(sleeps, [250]);
  assert.deepEqual(ops, ['sleep:start', 'sleep:end', 'send']);
  console.log('  ✓ a configured delay waits once, inside the queue, before the send');
}

// -- every queued send pays the delay; concurrency cannot skip it -----------
{
  // Two sends on one serialised queue: the order is sleep,send,sleep,send.
  // If the delay were applied outside the queue, a second handler enqueued
  // during the first wait would run without ever sleeping.
  const ops = [];
  let chain = Promise.resolve();
  const enqueueSend = (fn) => {
    const task = chain.then(() => fn(), () => fn());
    chain = task.catch(() => {});
    return task;
  };
  const throttled = createSendThrottle('10', {
    enqueueSend,
    sleep: async () => { ops.push('sleep'); },
  });
  await Promise.all([
    throttled(async () => { ops.push('send:1'); }),
    throttled(async () => { ops.push('send:2'); }),
  ]);
  assert.deepEqual(ops, ['sleep', 'send:1', 'sleep', 'send:2']);
  console.log('  ✓ concurrent handlers each pay the delay inside the serialised queue');
}

// -- a rejected send settles like the queue contract ------------------------
{
  let chain = Promise.resolve();
  const enqueueSend = (fn) => {
    const task = chain.then(() => fn(), () => fn());
    chain = task.catch(() => {});
    return task;
  };
  const throttled = createSendThrottle('1', { enqueueSend, sleep: async () => {} });
  await assert.rejects(
    throttled(async () => { throw new Error('send failed'); }),
    /send failed/,
  );
  console.log('  ✓ a send failure propagates to the awaiting HTTP handler');
}

console.log('bridge.throttle.test.mjs: all assertions passed');