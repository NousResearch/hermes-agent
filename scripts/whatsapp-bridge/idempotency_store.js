/**
 * Idempotency-Key store for the WhatsApp bridge.
 *
 * Provides deduplication for mutating HTTP routes (/send, /send-media,
 * /send-poll, /send-location, /edit).  Each entry is keyed by
 * (idempotencyKey, route, payloadHash); the first request executes
 * and caches status+body; repeats return the cached response without
 * re-invoking sock.sendMessage.
 *
 * Memory is bounded by:
 *   - TTL: entries expire after ttlMs (default 24h)
 *   - maxEntries: oldest entries are evicted when the limit is reached
 *
 * Concurrency: in-flight entries are tracked so concurrent identical
 * requests coalesce — all callers await a single execution promise.
 *
 * PII: only hashes are stored.  No payload bodies, chatIds, phone
 * numbers or tokens are kept in the store or in log output.
 *
 * Pure module — no Baileys/express side effects, safe for unit testing.
 */

import { createHash } from 'crypto';

const DEFAULT_TTL_MS = 24 * 60 * 60 * 1000; // 24 hours
const DEFAULT_MAX_ENTRIES = 10_000;

/**
 * @typedef {{ status: number, body: any }} CachedResponse
 */

/**
 * @typedef {{ status: number, body: any, conflict?: boolean }} IdempotencyResult
 */

/**
 * Create a new idempotency store.
 *
 * @param {object} [opts]
 * @param {number} [opts.ttlMs]       — entry TTL in ms (default 24h)
 * @param {number} [opts.maxEntries]  — maximum cached entries (default 10_000)
 * @returns {{ getOrExecute: Function, stats: Function, clear: Function }}
 */
export function createIdempotencyStore(opts = {}) {
  const ttlMs = opts.ttlMs ?? DEFAULT_TTL_MS;
  const maxEntries = opts.maxEntries ?? DEFAULT_MAX_ENTRIES;
  if (!Number.isFinite(ttlMs) || ttlMs <= 0) {
    throw new TypeError('ttlMs must be a positive finite number');
  }
  if (!Number.isInteger(maxEntries) || maxEntries <= 0) {
    throw new TypeError('maxEntries must be a positive integer');
  }

  // Map<string, entry> where key = `${idempotencyKey}:${route}:${payloadHash}`
  // We use a plain Map (insertion-ordered) so eviction = first key.
  /** @type {Map<string, Entry>} */
  const store = new Map();

  // Pending executions keyed identically — coalesce concurrent calls.
  /** @type {Map<string, Promise<CachedResponse>>} */
  const pending = new Map();

  // ---- helpers ----

  function compositeKey(idempotencyKey, route, payloadHash) {
    return `${idempotencyKey}\0${route}\0${payloadHash}`;
  }

  function isExpired(entry) {
    return Date.now() - entry.createdAt > ttlMs;
  }

  function evictIfNeeded() {
    while (store.size > maxEntries) {
      // Map preserves insertion order: first entry = oldest
      const oldestKey = store.keys().next().value;
      if (oldestKey === undefined) break;
      store.delete(oldestKey);
    }
  }

  function purgeExpired() {
    // Opportunistic sweep on every read — cheap because `size` is bounded.
    if (store.size === 0) return;
    const now = Date.now();
    for (const [key, entry] of store) {
      if (now - entry.createdAt > ttlMs) {
        store.delete(key);
      } else {
        // Map is insertion-ordered; once we hit a non-expired entry we could
        // stop, but TTLs vary and entries are not sorted by expiry.
      }
    }
  }

  // ---- public API ----

  /**
   * Execute `fn` if no cached result exists for the given triple, or
   * return the cached response.  Concurrent calls with the same triple
   * coalesce into a single `fn` execution.
   *
   * If the same idempotencyKey was seen before on the same route but
   * with a *different* payloadHash, returns a 409 conflict result without
   * calling `fn`.
   *
   * @param {string} idempotencyKey — raw Idempotency-Key header value
   * @param {string} route          — the HTTP route (e.g. '/send')
   * @param {string} payloadHash    — SHA-256 hash of the raw request body
   * @param {() => Promise<CachedResponse>} fn — the real operation
   * @returns {Promise<IdempotencyResult>}
   */
  async function getOrExecute(idempotencyKey, route, payloadHash, fn) {
    purgeExpired();

    const key = compositeKey(idempotencyKey, route, payloadHash);
    const sameKeyPrefix = `${idempotencyKey}\0${route}\0`;

    // Check for 409 conflict: same key+route, different payloadHash. Include
    // both completed and in-flight requests so concurrent conflicting payloads
    // cannot execute under the same idempotency key.
    for (const [existingKey, entry] of store) {
      if (existingKey.startsWith(sameKeyPrefix) && existingKey !== key) {
        if (isExpired(entry)) {
          store.delete(existingKey);
          break;
        }
        return {
          status: 409,
          body: { error: 'Idempotency-Key conflict: payload differs from the original request' },
          conflict: true,
        };
      }
    }
    for (const existingKey of pending.keys()) {
      if (existingKey.startsWith(sameKeyPrefix) && existingKey !== key) {
        return {
          status: 409,
          body: { error: 'Idempotency-Key conflict: payload differs from the original request' },
          conflict: true,
        };
      }
    }

    // Check for a cached completed response
    const cached = store.get(key);
    if (cached && !isExpired(cached)) {
      return { status: cached.status, body: cached.body };
    }
    // Expired entry — delete and re-execute
    if (cached) store.delete(key);

    // Check for a pending in-flight execution (coalesce)
    const inflight = pending.get(key);
    if (inflight) {
      return inflight;
    }

    // Start a new execution
    const execPromise = (async () => {
      try {
        const result = await fn();
        // Only cache successful executions (status 2xx). Explicit retryable
        // failures (429/502/503/504) must remain retryable by the Python client.
        if (result.status >= 200 && result.status < 300) {
          const entry = {
            status: result.status,
            body: result.body,
            createdAt: Date.now(),
          };
          store.set(key, entry);
          evictIfNeeded();
        }
        return { status: result.status, body: result.body };
      } finally {
        pending.delete(key);
      }
    })();

    pending.set(key, execPromise);
    return execPromise;
  }

  /**
   * Return stats about the store.  No PII is exposed.
   */
  function stats() {
    return {
      size: store.size + pending.size,
      maxEntries,
    };
  }

  /**
   * Remove all entries (useful for tests and graceful shutdown).
   */
  function clear() {
    store.clear();
    pending.clear();
  }

  return { getOrExecute, stats, clear };
}

/**
 * Hash a request body for idempotency comparison.
 *
 * Uses SHA-256. The body is JSON-serialised deterministically if it's
 * an object; raw strings are hashed as-is.
 *
 * @param {string|Buffer|object} body
 * @returns {string} hex digest (first 32 chars for compactness)
 */
export function hashPayload(body) {
  let data;
  if (Buffer.isBuffer(body)) {
    data = body;
  } else if (typeof body === 'string') {
    data = Buffer.from(body);
  } else {
    data = Buffer.from(JSON.stringify(body, Object.keys(body).sort()));
  }
  return createHash('sha256').update(data).digest('hex').slice(0, 32);
}
