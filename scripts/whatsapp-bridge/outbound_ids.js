/**
 * Bounded FIFO set of outbound message IDs.
 *
 * Used by the WhatsApp bridge to distinguish "echo of our own /send" from
 * "owner-typed message on the linked device" when forwarding `fromMe`
 * inbound events back to the Python adapter.
 *
 * Eviction drops the oldest insertion-order entry when the cap is exceeded.
 * Re-remembering an existing id is a no-op for ordering (not LRU refresh).
 *
 * With `persistPath`, the set is reloaded on startup and rewritten
 * atomically (tmp file + rename) whenever a new id is remembered. Without
 * it, a bridge restart empties the set and WhatsApp's redelivery of
 * pre-restart bot sends looks owner-typed to the owner-message gate.
 * A missing or unreadable file starts empty.
 */

import { readFileSync, renameSync, writeFileSync } from 'fs';

export function createOutboundIdTracker(maxSize = 512, { persistPath } = {}) {
  if (!Number.isInteger(maxSize) || maxSize < 1) {
    throw new RangeError('createOutboundIdTracker: maxSize must be a positive integer');
  }
  const ids = new Set();
  let warnedPersistFailure = false;

  function evict() {
    while (ids.size > maxSize) {
      // Set iteration order is insertion order, so values().next() is the
      // oldest entry — drop it to keep memory flat under sustained sending.
      ids.delete(ids.values().next().value);
    }
  }

  function load() {
    let saved;
    try {
      saved = JSON.parse(readFileSync(persistPath, 'utf8'));
    } catch {
      return;
    }
    if (!Array.isArray(saved)) return;
    for (const id of saved) {
      if (typeof id === 'string' && id) ids.add(id);
    }
    evict();
  }

  function save() {
    const tmpPath = `${persistPath}.tmp`;
    try {
      writeFileSync(tmpPath, JSON.stringify([...ids]));
      renameSync(tmpPath, persistPath);
    } catch (err) {
      if (!warnedPersistFailure) {
        warnedPersistFailure = true;
        console.warn(`[bridge] could not persist outbound message ids to ${persistPath}: ${err.message}`);
      }
    }
  }

  function remember(id) {
    if (!id || ids.has(id)) return;
    ids.add(id);
    evict();
    if (persistPath) save();
  }

  function has(id) {
    return Boolean(id) && ids.has(id);
  }

  function size() {
    return ids.size;
  }

  if (persistPath) load();

  return { remember, has, size };
}

/**
 * Pick the message id for an SDK send up front and remember it before the
 * call. Baileys can emit the `fromMe` upsert of our own send before
 * `sendMessage()` settles, and a send that times out here may still be
 * delivered; remembering the id only from the resolved result misses both.
 */
export function withRegisteredMessageId(tracker, options, generateId) {
  const messageId = options?.messageId || generateId();
  tracker.remember(messageId);
  return { ...options, messageId };
}
