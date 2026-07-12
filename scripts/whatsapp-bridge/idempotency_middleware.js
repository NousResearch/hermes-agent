/**
 * Idempotency-Key middleware for the WhatsApp bridge HTTP routes.
 *
 * Wraps mutating POST routes (/send, /send-media, /send-poll,
 * /send-location, /edit) with idempotency handling using the
 * pure `idempotency_store.js` module.
 *
 * Usage in bridge.js:
 *
 *   const { createIdempotencyStore, hashPayload } = from './idempotency_store.js';
 *   const { createIdempotencyMiddleware } = from './idempotency_middleware.js';
 *   const idempotencyStore = createIdempotencyStore(opts);
 *   const idempotencyMiddleware = createIdempotencyMiddleware(idempotencyStore);
 *
 *   app.post('/send', idempotencyMiddleware, async (req, res) => {
 *     // ... existing handler ...
 *   }, primeIdempotency(idempotencyStore));
 *
 * Or more flexibly, wrapping the handler body directly:
 *
 *   app.post('/send', async (req, res) => {
 *     const result = await idempotencyMiddleware.wrap(req, res, async () => {
 *       // ... existing handler body → return { status, body } ...
 *     });
 *     if (result) { res.status(result.status).json(result.body); }
 *   });
 *
 * Design:
 *   - No Idempotency-Key header → passthrough (executes normally)
 *   - GET/HEAD methods → passthrough (non-mutating)
 *   - Same key+route+payloadHash → returns cached response, no re-execution
 *   - Same key, different payload → 409 conflict
 *   - Concurrent identical → coalesced via pending promise
 *   - PII: only payload hash stored, never payload/chat/token
 *
 * Pure module — no Baileys/express side effects, safe for unit testing.
 */

import { hashPayload } from './idempotency_store.js';

/**
 * Mutating routes that participate in idempotency deduplication.
 */
const MUTATING_ROUTES = new Set([
  '/send',
  '/send-media',
  '/send-poll',
  '/send-location',
  '/edit',
]);

/**
 * Mutating HTTP methods.
 */
const MUTATING_METHODS = new Set(['POST', 'PUT', 'PATCH']);

/**
 * Create the idempotency middleware. The middleware is a function that
 * can be used in two ways:
 *
 * 1. Express middleware: `app.post('/send', idemMiddleware, handler)`
 *    — but since we need the result to short-circuit, we also expose
 *    `wrap()` for manual wrapping.
 *
 * 2. Programmatic: `await idemMiddleware.wrap(req, res, fn)` where
 *    `fn` is `async () => ({ status, body })`.
 *
 * @param {ReturnType<typeof import('./idempotency_store.js').createIdempotencyStore>} store
 */
export function createIdempotencyMiddleware(store) {

  /**
   * Check if a request should participate in idempotency:
   *   - Must be a mutating method (POST/PUT/PATCH)
   *   - Must have an Idempotency-Key header
   *   - Route must be one of the known mutating routes
   */
  function shouldDedup(req) {
    if (!req || !req.method || !MUTATING_METHODS.has(req.method)) return false;
    if (!req.headers || !(req.headers['idempotency-key'] || req.headers['Idempotency-Key'])) return false;
    if (!MUTATING_ROUTES.has(req.path)) return false;
    return true;
  }

  /**
   * Programmatic wrapper. `(req, res, execute)` → { status, body }
   *   - null when no Idempotency-Key header: caller proceeds normally
   *   - { status, body } when cached/conflict: caller should respond
   *
   * The middleware object is callable:
   *   const middleware = createIdempotencyMiddleware(store);
   *   const result = await middleware(req, res, execute);
   *   // semi(null → proceed; object → respond)
   *
   * Also exposes `.wrap()` as alias for explicit use.
   */
  async function wrap(req, res, execute) {
    if (!shouldDedup(req)) {
      const direct = await execute();
      res.status(direct.status).json(direct.body);
      return direct;
    }

    const key = req.headers['idempotency-key'] || req.headers['Idempotency-Key'];
    const route = req.path;
    const payloadHash = hashPayload(req.body);
    const result = await store.getOrExecute(key, route, payloadHash, execute);
    res.status(result.status).json(result.body);
    return result;
  }

  /** Express middleware that captures the first JSON response and replays it
   * for duplicate requests. Existing route handlers remain unchanged. */
  async function expressMiddleware(req, res, next) {
    if (!shouldDedup(req)) return next();

    const key = req.headers['idempotency-key'] || req.headers['Idempotency-Key'];
    const route = req.path;
    const payloadHash = hashPayload(req.body);
    const originalStatus = res.status.bind(res);
    const originalJson = res.json.bind(res);

    try {
      const result = await store.getOrExecute(key, route, payloadHash, () =>
        new Promise((resolve, reject) => {
          let status = 200;
          res.status = (code) => { status = code; return res; };
          res.json = (body) => { resolve({ status, body }); return res; };
          try {
            const maybePromise = next();
            if (maybePromise && typeof maybePromise.catch === 'function') {
              maybePromise.catch(reject);
            }
          } catch (err) {
            reject(err);
          }
        })
      );
      return originalStatus(result.status).json(result.body);
    } catch (err) {
      res.status = originalStatus;
      res.json = originalJson;
      return next(err);
    }
  }

  // Programmatic callable used by pure tests; Express uses `.express`.
  const middleware = wrap;
  middleware.wrap = wrap;
  middleware.express = expressMiddleware;
  middleware.shouldDedup = shouldDedup;

  return middleware;
}
