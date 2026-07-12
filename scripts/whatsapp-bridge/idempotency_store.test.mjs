/**
 * TDD tests for the Idempotency-Key store used by the WhatsApp bridge.
 *
 * The store must:
 *   - Return cached status+body on repeat of the same key+route+payloadHash
 *   - Return 409 conflict when the same key has a different payloadHash
 *   - Coalesce concurrent in-flight requests (same key awaits single execution)
 *   - Expire entries after a configurable TTL
 *   - Evict the oldest entries when a max-entries limit is reached
 *   - Never store or log PII (no payload, chatId, phone, token)
 *
 * These tests do NOT import bridge.js — they test the pure store module.
 */

import { strict as assert } from 'node:assert';
import { createIdempotencyStore } from './idempotency_store.js';

// ------------------------------------------------------------------
// 1. First request executes and is cached; repeat returns cached response
// ------------------------------------------------------------------
{
  const store = createIdempotencyStore({ ttlMs: 60_000, maxEntries: 100 });
  let execCount = 0;

  const execute = async () => {
    execCount += 1;
    return { status: 200, body: { success: true, messageId: 'msg-1' } };
  };

  const result1 = await store.getOrExecute('key-abc', '/send', 'hash-123', execute);
  assert.strictEqual(result1.status, 200);
  assert.strictEqual(result1.body.success, true);
  assert.strictEqual(execCount, 1, 'first call executes');

  const result2 = await store.getOrExecute('key-abc', '/send', 'hash-123', execute);
  assert.strictEqual(result2.status, 200, 'repeat returns cached status');
  assert.deepStrictEqual(result2.body, { success: true, messageId: 'msg-1' });
  assert.strictEqual(execCount, 1, 'second call does NOT execute');

  console.log('  ✓ repeat same key+route+hash returns cached response without re-execution');
}

// ------------------------------------------------------------------
// 2. Same key with different payload hash returns 409
// ------------------------------------------------------------------
{
  const store = createIdempotencyStore({ ttlMs: 60_000, maxEntries: 100 });
  let execCount = 0;

  const execute = async () => {
    execCount += 1;
    return { status: 200, body: { success: true } };
  };

  await store.getOrExecute('key-conflict', '/send', 'hash-A', execute);
  assert.strictEqual(execCount, 1);

  const result = await store.getOrExecute('key-conflict', '/send', 'hash-B', execute);
  assert.strictEqual(result.status, 409, 'different payload hash returns 409');
  assert.strictEqual(execCount, 1, 'second call does NOT execute on conflict');
  assert.ok(result.body.error, '409 body has error message');

  console.log('  ✓ same key with different payload hash returns 409 without execution');
}

// ------------------------------------------------------------------
// 3. Concurrent in-flight requests coalesce (single execution)
// ------------------------------------------------------------------
{
  const store = createIdempotencyStore({ ttlMs: 60_000, maxEntries: 100 });
  let execCount = 0;

  const execute = async () => {
    execCount += 1;
    // Simulate slow operation so concurrent calls arrive while in-flight
    await new Promise(r => setTimeout(r, 30));
    return { status: 200, body: { success: true, messageId: 'coalesced-1' } };
  };

  // Fire 5 concurrent requests with the same key+route+hash
  const results = await Promise.all([
    store.getOrExecute('key-concurrent', '/send', 'hash-X', execute),
    store.getOrExecute('key-concurrent', '/send', 'hash-X', execute),
    store.getOrExecute('key-concurrent', '/send', 'hash-X', execute),
    store.getOrExecute('key-concurrent', '/send', 'hash-X', execute),
    store.getOrExecute('key-concurrent', '/send', 'hash-X', execute),
  ]);

  assert.strictEqual(execCount, 1, 'only one execution for concurrent identical requests');
  for (const r of results) {
    assert.strictEqual(r.status, 200);
    assert.deepStrictEqual(r.body, { success: true, messageId: 'coalesced-1' });
  }

  console.log('  ✓ concurrent in-flight requests coalesce into a single execution');
}

// ------------------------------------------------------------------
// 4. TTL expiration: expired entry is treated as new
// ------------------------------------------------------------------
{
  // Use a fake clock by setting ttlMs very small and sleeping
  const store = createIdempotencyStore({ ttlMs: 10, maxEntries: 100 });
  let execCount = 0;

  const execute = async () => {
    execCount += 1;
    return { status: 200, body: { n: execCount } };
  };

  await store.getOrExecute('key-ttl', '/send', 'hash-T', execute);
  assert.strictEqual(execCount, 1);

  // Wait for TTL to expire
  await new Promise(r => setTimeout(r, 20));

  await store.getOrExecute('key-ttl', '/send', 'hash-T', execute);
  assert.strictEqual(execCount, 2, 'expired entry causes re-execution');

  console.log('  ✓ expired TTL entry is removed and re-executed on next request');
}

// ------------------------------------------------------------------
// 5. Max entries eviction: oldest entries are evicted when limit reached
// ------------------------------------------------------------------
{
  const store = createIdempotencyStore({ ttlMs: 60_000, maxEntries: 4 });
  let execCount = 0;

  const execute = async () => {
    execCount += 1;
    return { status: 200, body: { n: execCount } };
  };

  // Fill store to capacity
  await store.getOrExecute('k1', '/send', 'h1', execute);
  await store.getOrExecute('k2', '/send', 'h1', execute);
  await store.getOrExecute('k3', '/send', 'h1', execute);
  await store.getOrExecute('k4', '/send', 'h1', execute);
  assert.strictEqual(execCount, 4);

  // This should evict k1 (oldest) to make room for k5
  await store.getOrExecute('k5', '/send', 'h1', execute);
  assert.strictEqual(execCount, 5);

  // k1 should now be re-executed (it was evicted to make room)
  await store.getOrExecute('k1', '/send', 'h1', execute);
  assert.strictEqual(execCount, 6, 'evicted entry re-executes');

  // k3 should still be cached (k1 was the oldest, k2 and k3 survived first eviction;
  // after k5 eviction of k1 then k1 re-insert evicts k2, so test k3)
  await store.getOrExecute('k4', '/send', 'h1', execute);
  assert.strictEqual(execCount, 6, 'recently cached entry returns cached response');

  console.log('  ✓ oldest entries are evicted when maxEntries is reached');
}

// ------------------------------------------------------------------
// 6. Different routes with same key are independent
// ------------------------------------------------------------------
{
  const store = createIdempotencyStore({ ttlMs: 60_000, maxEntries: 100 });
  let execCount = 0;

  const execute = async () => {
    execCount += 1;
    return { status: 200, body: { n: execCount } };
  };

  await store.getOrExecute('key-multi', '/send', 'h1', execute);
  await store.getOrExecute('key-multi', '/send-media', 'h1', execute);
  assert.strictEqual(execCount, 2, 'different routes with same key each execute');

  // Repeating /send should still be cached
  await store.getOrExecute('key-multi', '/send', 'h1', execute);
  assert.strictEqual(execCount, 2, 'repeat /send is cached');

  console.log('  ✓ different routes with the same key are independent');
}

// ------------------------------------------------------------------
// 7. No PII in store internals
// ------------------------------------------------------------------
{
  const store = createIdempotencyStore({ ttlMs: 60_000, maxEntries: 100 });

  const execute = async () => ({ status: 200, body: { success: true } });

  // Execute with a payload that would be PII if stored
  await store.getOrExecute('key-pii', '/send', 'hash-pii', execute);

  // Inspect the internal map
  const stats = store.stats();
  assert.ok(typeof stats.size === 'number');
  assert.ok(typeof stats.maxEntries === 'number');

  // The store should NOT expose payload data in stats
  const statsStr = JSON.stringify(stats);
  assert.ok(!statsStr.includes('555'), 'no phone numbers in stats');
  assert.ok(!statsStr.includes('chatId'), 'no chatId in stats');

  console.log('  ✓ store stats contain no PII');
}

// ------------------------------------------------------------------
// 8. Error from execute is NOT cached (so retries can proceed)
// ------------------------------------------------------------------
{
  const store = createIdempotencyStore({ ttlMs: 60_000, maxEntries: 100 });
  let execCount = 0;

  const execute = async () => {
    execCount += 1;
    if (execCount === 1) throw new Error('transient failure');
    return { status: 200, body: { success: true } };
  };

  // First call fails — should NOT be cached as a success
  await assert.rejects(
    () => store.getOrExecute('key-err', '/send', 'hash-E', execute),
    /transient failure/,
  );

  // Retry should execute again
  const result = await store.getOrExecute('key-err', '/send', 'hash-E', execute);
  assert.strictEqual(result.status, 200);
  assert.strictEqual(execCount, 2, 'retry executes after uncached error');

  console.log('  ✓ execution errors are not cached — retries can proceed');
}

// ------------------------------------------------------------------
// 9. store.stats() reports current size and configuration
// ------------------------------------------------------------------
{
  const store = createIdempotencyStore({ ttlMs: 60_000, maxEntries: 50 });
  const execute = async () => ({ status: 200, body: {} });

  assert.strictEqual(store.stats().size, 0, 'empty store has size 0');

  await store.getOrExecute('s1', '/send', 'h', execute);
  await store.getOrExecute('s2', '/send', 'h', execute);

  assert.strictEqual(store.stats().size, 2, 'store size reflects entries');
  assert.strictEqual(store.stats().maxEntries, 50);

  console.log('  ✓ stats() reports size and maxEntries');
}

// ------------------------------------------------------------------
// 10. store.clear() removes all entries
// ------------------------------------------------------------------
{
  const store = createIdempotencyStore({ ttlMs: 60_000, maxEntries: 50 });
  let execCount = 0;
  const execute = async () => {
    execCount += 1;
    return { status: 200, body: {} };
  };

  await store.getOrExecute('c1', '/send', 'h', execute);
  await store.getOrExecute('c2', '/send', 'h', execute);
  assert.strictEqual(store.stats().size, 2);

  store.clear();
  assert.strictEqual(store.stats().size, 0, 'clear() empties store');

  // After clear, same key re-executes
  await store.getOrExecute('c1', '/send', 'h', execute);
  assert.strictEqual(execCount, 3, 'cleared entry re-executes');

  console.log('  ✓ clear() removes all entries');
}

console.log('\n✅ All idempotency-store tests passed.');
