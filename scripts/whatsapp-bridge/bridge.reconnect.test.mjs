/**
 * Unit tests for the reconnect scheduling and version resolution guards.
 *
 * Regression tests for the reconnect-wedge trap: startSocket() awaits
 * network I/O (fetchLatestBaileysVersion has no AbortSignal) before it
 * creates a socket, and the close handler used to re-enter it via a bare
 * `setTimeout(startSocket, ...)`. A rejection was unhandled and a stalled
 * fetch left the bridge permanently disconnected while its HTTP server
 * kept answering 503 — observed in the field as a bridge that logged
 * "Reconnecting in 3s..." once and then went silent for 27+ hours.
 *
 * These tests avoid importing bridge.js because that file starts an HTTP
 * server and Baileys socket at module load. Keep the helper module pure.
 */

import { strict as assert } from 'node:assert';

import {
  createReconnectScheduler,
  createVersionResolver,
} from './bridge_helpers.js';

const tick = () => new Promise(resolve => setImmediate(resolve));
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));

// -- createReconnectScheduler ---------------------------------------------

// A rejecting start function is caught and rescheduled at the retry delay;
// a subsequent success stops the retry chain.
{
  const timers = [];
  const logs = [];
  let attempts = 0;
  const startFn = async () => {
    attempts += 1;
    if (attempts === 1) throw new Error('boom');
  };

  const schedule = createReconnectScheduler(startFn, {
    retryDelayMs: 5000,
    log: line => logs.push(line),
    setTimeoutFn: (fn, ms) => timers.push({ fn, ms }),
  });

  schedule(3000);
  assert.equal(timers.length, 1);
  assert.equal(timers[0].ms, 3000);

  timers[0].fn();
  await tick();
  await tick();

  assert.equal(attempts, 1);
  assert.equal(logs.length, 1);
  assert.match(logs[0], /Reconnect failed \(boom\)/);
  assert.equal(timers.length, 2, 'rejection must schedule a retry');
  assert.equal(timers[1].ms, 5000);

  timers[1].fn();
  await tick();
  await tick();

  assert.equal(attempts, 2);
  assert.equal(timers.length, 2, 'success must not schedule another attempt');
  assert.equal(logs.length, 1);
}

// A synchronous throw from the start function is contained the same way as
// an async rejection.
{
  const timers = [];
  const logs = [];
  const schedule = createReconnectScheduler(
    () => { throw new Error('sync boom'); },
    {
      retryDelayMs: 1000,
      log: line => logs.push(line),
      setTimeoutFn: (fn, ms) => timers.push({ fn, ms }),
    },
  );

  schedule(0);
  timers[0].fn();
  await tick();
  await tick();

  assert.equal(logs.length, 1);
  assert.match(logs[0], /sync boom/);
  assert.equal(timers.length, 2);
}

// Consecutive attempts back off exponentially and stop at the cap. Without
// this a persistent failure (unreachable proxy, pre-auth 428/503) reconnected
// every 3-5s indefinitely, because each close scheduled a fresh fixed delay.
{
  const timers = [];
  const schedule = createReconnectScheduler(async () => {}, {
    backoffBaseMs: 3000,
    maxDelayMs: 60000,
    randomFn: () => 0.5,          // midpoint => jitter contributes nothing
    log: () => {},
    setTimeoutFn: (fn, ms) => timers.push({ fn, ms }),
  });

  const waits = [];
  for (let i = 0; i < 8; i += 1) waits.push(schedule(3000));

  // First two honour the caller's delay so a single blip recovers promptly.
  assert.deepEqual(waits.slice(0, 2), [3000, 3000]);
  assert.deepEqual(waits.slice(2, 6), [6000, 12000, 24000, 48000]);
  assert.deepEqual(waits.slice(6), [60000, 60000], 'must settle at the cap');
  assert.deepEqual(timers.map(t => t.ms), waits, 'scheduled delay must match');
}

// The cap bounds the value actually scheduled, not just the pre-jitter one.
// Jittering after the clamp would overshoot by jitterRatio -- a "5 minute"
// cap that really returned 6 minutes.
{
  const schedule = createReconnectScheduler(async () => {}, {
    backoffBaseMs: 3000,
    maxDelayMs: 60000,
    jitterRatio: 0.2,
    randomFn: () => 1,            // maximum jitter
    log: () => {},
    setTimeoutFn: () => {},
  });

  const waits = [];
  for (let i = 0; i < 12; i += 1) waits.push(schedule(3000));

  assert.ok(waits.every(w => w <= 60000), `cap breached: ${waits.join(',')}`);
  assert.equal(waits.at(-1), 60000);
}

// Escalation is rooted in one fixed base, not the caller's delay. The close
// handler passes 1000 for a 515 restart and 3000 otherwise, and the retry
// path passes 5000; rooting the curve in the request would give three
// different curves sharing one counter, so interleaved calls could step
// backwards. A request of 0 would also stay 0 forever.
{
  const schedule = createReconnectScheduler(async () => {}, {
    backoffBaseMs: 3000,
    randomFn: () => 0.5,
    log: () => {},
    setTimeoutFn: () => {},
  });

  const waits = [schedule(3000), schedule(1000), schedule(5000), schedule(1000), schedule(0)];

  // Past the two warm-up attempts the sequence ignores the requested value
  // and climbs monotonically.
  const escalated = waits.slice(2);
  assert.deepEqual(escalated, [6000, 12000, 24000]);
  assert.ok(
    escalated.every((v, i, a) => i === 0 || v > a[i - 1]),
    'delays must not step backwards when call sites interleave',
  );
  assert.ok(waits.at(-1) > 0, 'a base of 0 must not bypass backoff');
}

// A healthy connection resets the backoff, so an unrelated later drop does
// not inherit the previous outage's delay.
{
  const schedule = createReconnectScheduler(async () => {}, {
    backoffBaseMs: 3000,
    randomFn: () => 0.5,
    log: () => {},
    setTimeoutFn: () => {},
  });

  for (let i = 0; i < 5; i += 1) schedule(3000);
  assert.ok(schedule(3000) > 3000, 'backoff should have grown');

  schedule.reset();
  assert.equal(schedule(3000), 3000, 'reset must return to the caller delay');
}

// The scheduler stays a plain callable so existing call sites are unaffected.
{
  const schedule = createReconnectScheduler(async () => {}, {
    log: () => {},
    setTimeoutFn: () => {},
  });
  assert.equal(typeof schedule, 'function');
  assert.equal(typeof schedule.reset, 'function');
}

// Jitter spreads retries around the backed-off delay rather than having every
// bridge recovering from one outage retry in lockstep.
{
  const mk = random => {
    const schedule = createReconnectScheduler(async () => {}, {
      backoffBaseMs: 1000,
      jitterRatio: 0.2,
      randomFn: () => random,
      log: () => {},
      setTimeoutFn: () => {},
    });
    for (let i = 0; i < 3; i += 1) schedule(1000);
    return schedule(1000);
  };

  const low = mk(0);
  const mid = mk(0.5);
  const high = mk(1);

  assert.ok(low < mid && mid < high, 'jitter must vary with the RNG');
  assert.ok(low >= mid * 0.8 - 1 && high <= mid * 1.2 + 1, 'jitter stays within +/-20%');
}

// Jitter never perturbs the first two attempts -- the close handler relies on
// the exact 1s/3s delays it asks for.
{
  const schedule = createReconnectScheduler(async () => {}, {
    randomFn: () => 1,            // maximum jitter, if it applied
    log: () => {},
    setTimeoutFn: () => {},
  });

  assert.equal(schedule(3000), 3000);
  assert.equal(schedule(1000), 1000);
}

// -- createVersionResolver ------------------------------------------------

// A successful fetch returns and caches the version.
{
  const resolveVersion = createVersionResolver(
    async () => ({ version: [2, 3000, 99] }),
    { log: () => {} },
  );
  assert.deepEqual(await resolveVersion(), [2, 3000, 99]);
}

// A fetch that never settles resolves within the timeout bound instead of
// pending forever; before any success there is no cache, so the resolver
// yields null (callers fall back to the Baileys default).
{
  const logs = [];
  const resolveVersion = createVersionResolver(
    () => new Promise(() => {}),
    { timeoutMs: 20, log: line => logs.push(line) },
  );
  assert.equal(await resolveVersion(), null);
  assert.equal(logs.length, 1);
  assert.match(logs[0], /version fetch timed out/);
  assert.match(logs[0], /library default/);
}

// After one success, later failures fall back to the cached version.
{
  const logs = [];
  let calls = 0;
  const resolveVersion = createVersionResolver(
    async () => {
      calls += 1;
      if (calls === 1) return { version: [2, 3000, 42] };
      throw new Error('network down');
    },
    { timeoutMs: 20, log: line => logs.push(line) },
  );
  assert.deepEqual(await resolveVersion(), [2, 3000, 42]);
  assert.deepEqual(await resolveVersion(), [2, 3000, 42]);
  assert.equal(logs.length, 1);
  assert.match(logs[0], /network down/);
  assert.match(logs[0], /cached version/);
}

// The losing timeout timer is cleared after a fast success, so the resolver
// does not hold the event loop open for the full timeout window.
{
  const resolveVersion = createVersionResolver(
    async () => ({ version: [2, 3000, 1] }),
    { timeoutMs: 60_000, log: () => {} },
  );
  const before = Date.now();
  await resolveVersion();
  await sleep(10);
  assert.ok(Date.now() - before < 1000);
}

console.log('bridge.reconnect.test.mjs: all assertions passed');
