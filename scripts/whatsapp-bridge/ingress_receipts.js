/**
 * Persistent, inter-process-safe ingress receipts for the WhatsApp bridge.
 *
 * Baileys can redeliver the same inbound message many times: connection
 * flaps, history sync after a re-pair, and `messages.upsert` replays all
 * produce events whose `(remoteJid, key.id)` pair was already processed.
 * The in-memory echo filter (`outbound_ids.js`) only covers our own sends
 * and does not survive a bridge restart, so a restart in the middle of a
 * redelivery storm forwards every duplicate to the Python adapter — and
 * from there each one becomes a model-facing event with real token cost.
 * The incident that motivated this involved 24 concurrent bridge
 * processes, so the claim must be atomic ACROSS processes, not just
 * within one.
 *
 * Storage is one empty claim file per `(chatId, messageId)` inside a
 * receipts directory. The claim is an exclusive create (`O_CREAT|O_EXCL`),
 * which the kernel arbitrates: when several bridge processes race on the
 * same message, exactly one create succeeds and every other process sees
 * `EEXIST` and drops its copy. The directory lives OUTSIDE the session dir
 * (deleting the session to re-pair must not erase receipts, because
 * re-pairing is exactly when history replay happens). Past `maxEntries`
 * claims, the directory is compacted down to the newest `compactTo` by
 * file mtime.
 *
 * Every storage failure fails open: the message is forwarded and the key
 * is NOT remembered, so a later replay is only dropped once a durable
 * receipt actually exists. A rare duplicate is cheaper than silently
 * dropping real messages while the disk misbehaves.
 */

import { closeSync, existsSync, mkdirSync, openSync, readdirSync, statSync, unlinkSync } from 'fs';
import path from 'path';

const KEY_SEPARATOR = ' ';

export function receiptKey(chatId, messageId) {
  return `${chatId}${KEY_SEPARATOR}${messageId}`;
}

/** Claim files are the encoded key; `seen` stores encoded names only. */
function claimName(chatId, messageId) {
  return encodeURIComponent(receiptKey(chatId, messageId));
}

export function createIngressReceipts({
  dirPath,
  maxEntries = 50000,
  compactTo = 25000,
  logger = console,
} = {}) {
  if (!dirPath) throw new Error('ingress receipts: dirPath is required');
  if (compactTo >= maxEntries) throw new Error('ingress receipts: compactTo must be < maxEntries');

  // Fast path only; the durable claim below is what other processes see.
  let seen = new Set();

  function warn(message, err) {
    logger.warn?.(`[bridge] ${message}: ${err.message}`);
  }

  function load() {
    seen = new Set();
    try {
      mkdirSync(dirPath, { recursive: true });
      for (const name of readdirSync(dirPath)) seen.add(name);
    } catch (err) {
      // A damaged receipts dir must not take the bridge down; running
      // without receipts only risks duplicate delivery, never message loss.
      warn('ingress receipts dir unusable, deduplication degraded', err);
    }
  }

  function compact() {
    try {
      const entries = [];
      for (const name of readdirSync(dirPath)) {
        try {
          entries.push([name, statSync(path.join(dirPath, name)).mtimeMs]);
        } catch {
          // Removed by a concurrent compaction; already gone is fine.
        }
      }
      entries.sort((a, b) => a[1] - b[1]);
      for (const [name] of entries.slice(0, Math.max(0, entries.length - compactTo))) {
        try {
          unlinkSync(path.join(dirPath, name));
        } catch {
          // Same: losing the race to delete an old claim is harmless.
        }
      }
      seen = new Set(entries.slice(-compactTo).map(([name]) => name));
    } catch (err) {
      warn('ingress receipts compaction failed, keeping current receipts', err);
    }
  }

  load();
  if (seen.size > maxEntries) compact();

  return {
    /** True when this (chatId, messageId) was already recorded. */
    has(chatId, messageId) {
      const name = claimName(chatId, messageId);
      return seen.has(name) || existsSync(path.join(dirPath, name));
    },

    /**
     * Claim a message. Returns true when this call won the claim (caller
     * should process the message) and false when it is a duplicate (caller
     * should drop it). The claim is durable and atomic across processes:
     * it exists on disk before this returns true.
     */
    record(chatId, messageId) {
      const name = claimName(chatId, messageId);
      if (seen.has(name)) return false;
      const claimPath = path.join(dirPath, name);
      try {
        try {
          closeSync(openSync(claimPath, 'wx'));
        } catch (err) {
          if (err.code !== 'ENOENT') throw err;
          // The receipts dir vanished or never got created; one recreate
          // attempt so a transient gap does not disable dedup for good.
          mkdirSync(dirPath, { recursive: true });
          closeSync(openSync(claimPath, 'wx'));
        }
      } catch (err) {
        if (err.code === 'EEXIST') {
          // Another process (or an earlier run) holds the claim.
          seen.add(name);
          return false;
        }
        // Fail open without remembering the key: a key retained here would
        // silently drop replays that no durable receipt ever covered.
        warn('ingress receipt write failed, forwarding without dedup', err);
        return true;
      }
      seen.add(name);
      if (seen.size > maxEntries) compact();
      return true;
    },

    /** Number of receipts currently tracked (for tests/diagnostics). */
    size() {
      return seen.size;
    },
  };
}
