import path from 'node:path';
import {
  closeSync, fsyncSync, mkdirSync, openSync, readFileSync, renameSync, writeFileSync,
} from 'node:fs';
import { randomBytes } from 'node:crypto';

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
 * Heuristic limitation (intentional, documented for future debugging):
 * the set is in-memory only.  On bridge restart it is empty, so for the
 * brief window between restart and the first new outbound, any in-flight
 * delivery receipts of pre-restart sends would be classified as
 * owner-typed.  The TTL on owner-driven plugin actions (e.g. handover
 * sliding TTL) bounds blast radius; persisting would not be worth the
 * extra complexity / disk churn.
 */

export function createOutboundIdTracker(maxSize = 512) {
  if (!Number.isInteger(maxSize) || maxSize < 1) {
    throw new RangeError('createOutboundIdTracker: maxSize must be a positive integer');
  }
  const ids = new Set();

  function remember(id) {
    if (!id) return;
    ids.add(id);
    while (ids.size > maxSize) {
      // Set iteration order is insertion order, so values().next() is the
      // oldest entry — drop it to keep memory flat under sustained sending.
      ids.delete(ids.values().next().value);
    }
  }

  function has(id) {
    return Boolean(id) && ids.has(id);
  }

  function size() {
    return ids.size;
  }

  return { remember, has, size };
}

export function createDurableOutboundIdTracker({
  directory,
  maxSize = 10000,
  retentionMs = 30 * 24 * 60 * 60 * 1000,
  now = Date.now,
} = {}) {
  if (typeof directory !== 'string' || !directory) {
    throw new TypeError('directory is required for durable outbound provenance');
  }
  if (!Number.isInteger(maxSize) || maxSize < 1) {
    throw new RangeError('maxSize must be a positive integer');
  }
  if (!Number.isFinite(retentionMs) || retentionMs < 1) {
    throw new RangeError('retentionMs must be positive');
  }
  if (typeof now !== 'function') throw new TypeError('now must be a function');
  mkdirSync(directory, { recursive: true });
  const statePath = path.join(directory, 'connector-outbound-ids.json');
  const ids = new Map();
  let journalEntries = 0;
  let needsMigration = false;

  function syncDirectory() {
    let descriptor;
    try {
      descriptor = openSync(directory, 'r');
      fsyncSync(descriptor);
    } finally {
      if (descriptor !== undefined) closeSync(descriptor);
    }
  }

  function compact() {
    const temporaryPath = `${statePath}.${process.pid}.${randomBytes(6).toString('hex')}.tmp`;
    const descriptor = openSync(temporaryPath, 'w', 0o600);
    try {
      for (const [id, rememberedAt] of ids) {
        writeFileSync(descriptor, `${JSON.stringify({ id, rememberedAt })}\n`, { encoding: 'utf8' });
      }
      fsyncSync(descriptor);
    } finally {
      closeSync(descriptor);
    }
    renameSync(temporaryPath, statePath);
    syncDirectory();
    journalEntries = ids.size;
  }

  function append(id, rememberedAt) {
    const descriptor = openSync(statePath, 'a', 0o600);
    try {
      writeFileSync(descriptor, `${JSON.stringify({ id, rememberedAt })}\n`, { encoding: 'utf8' });
      fsyncSync(descriptor);
    } finally {
      closeSync(descriptor);
    }
    syncDirectory();
    journalEntries += 1;
  }

  function pruneExpired(referenceTime) {
    const cutoff = referenceTime - retentionMs;
    let removed = false;
    for (const [id, rememberedAt] of ids) {
      if (rememberedAt <= cutoff) {
        ids.delete(id);
        removed = true;
      }
    }
    return removed;
  }

  try {
    const raw = readFileSync(statePath, 'utf8');
    try {
      const legacy = JSON.parse(raw);
      if (Array.isArray(legacy?.ids)) {
        for (const id of legacy.ids) {
          if (typeof id === 'string' && id) ids.set(id, now());
        }
        needsMigration = true;
      } else if (typeof legacy?.id === 'string' && Number.isFinite(legacy.rememberedAt)) {
        ids.set(legacy.id, legacy.rememberedAt);
        journalEntries = 1;
      }
    } catch {
      const lines = raw.split('\n');
      for (let index = 0; index < lines.length; index += 1) {
        const line = lines[index];
        if (!line) continue;
        let entry;
        try {
          entry = JSON.parse(line);
        } catch (error) {
          const isIncompleteTail = index === lines.length - 1 && !raw.endsWith('\n');
          if (!isIncompleteTail) throw error;
          needsMigration = true;
          break;
        }
        if (typeof entry?.id === 'string' && entry.id && Number.isFinite(entry.rememberedAt)) {
          ids.set(entry.id, entry.rememberedAt);
          journalEntries += 1;
        }
      }
    }
  } catch (error) {
    if (error?.code !== 'ENOENT') throw error;
  }
  const removedAtStartup = pruneExpired(now());
  if (needsMigration || removedAtStartup) compact();

  return {
    remember(id) {
      if (typeof id !== 'string' || !id || ids.has(id)) return;
      const rememberedAt = now();
      const removed = pruneExpired(rememberedAt);
      ids.set(id, rememberedAt);
      append(id, rememberedAt);
      if (removed || (journalEntries > maxSize * 2 && journalEntries > ids.size * 2)) compact();
    },
    has(id) {
      if (typeof id !== 'string') return false;
      pruneExpired(now());
      return ids.has(id);
    },
    size() {
      pruneExpired(now());
      return ids.size;
    },
  };
}

export async function sendWithProvenance({ tracker, messageId, send }) {
  if (!tracker || typeof tracker.remember !== 'function') throw new TypeError('tracker is required');
  if (typeof messageId !== 'string' || !messageId) throw new TypeError('messageId is required');
  if (typeof send !== 'function') throw new TypeError('send is required');
  tracker.remember(messageId);
  return send(messageId);
}
