/**
 * Persistent ingress receipts for the WhatsApp bridge.
 *
 * Baileys can redeliver the same inbound message many times: connection
 * flaps, history sync after a re-pair, and `messages.upsert` replays all
 * produce events whose `(remoteJid, key.id)` pair was already processed.
 * The in-memory echo filter (`outbound_ids.js`) only covers our own sends
 * and does not survive a bridge restart, so a restart in the middle of a
 * redelivery storm forwards every duplicate to the Python adapter — and
 * from there each one becomes a model-facing event with real token cost.
 *
 * This tracker records a durable receipt per `(chatId, messageId)` BEFORE
 * the message is processed. The receipt is appended to disk first and only
 * then added to the in-memory set, so a crash between the two never lets a
 * message through twice. Duplicates are dropped at the bridge door without
 * media downloads or queue traffic.
 *
 * Storage is a newline-delimited key log next to the session directory
 * (deliberately OUTSIDE it: deleting the session to re-pair must not erase
 * receipts, because re-pairing is exactly when history replay happens).
 * When the log exceeds `maxEntries` it is compacted in place to the newest
 * `compactTo` keys via a tmp-file + atomic rename.
 */

import { appendFileSync, existsSync, mkdirSync, readFileSync, renameSync, writeFileSync } from 'fs';
import path from 'path';

const KEY_SEPARATOR = ' ';

export function receiptKey(chatId, messageId) {
  return `${chatId}${KEY_SEPARATOR}${messageId}`;
}

export function createIngressReceipts({
  filePath,
  maxEntries = 50000,
  compactTo = 25000,
  logger = console,
} = {}) {
  if (!filePath) throw new Error('ingress receipts: filePath is required');
  if (compactTo >= maxEntries) throw new Error('ingress receipts: compactTo must be < maxEntries');

  // Insertion-ordered by construction; JS Sets iterate in insertion order,
  // which is what compaction relies on to keep the newest keys.
  let seen = new Set();

  function load() {
    seen = new Set();
    if (!existsSync(filePath)) return;
    try {
      for (const line of readFileSync(filePath, 'utf8').split('\n')) {
        if (line) seen.add(line);
      }
    } catch (err) {
      // A damaged receipts file must not take the bridge down; starting
      // empty only risks duplicate delivery, never message loss.
      logger.warn?.(`[bridge] ingress receipts unreadable, starting empty: ${err.message}`);
      seen = new Set();
    }
  }

  function compact() {
    const keep = [...seen].slice(-compactTo);
    const tmpPath = `${filePath}.tmp`;
    writeFileSync(tmpPath, keep.length ? keep.join('\n') + '\n' : '');
    renameSync(tmpPath, filePath);
    seen = new Set(keep);
  }

  mkdirSync(path.dirname(filePath), { recursive: true });
  load();
  if (seen.size > maxEntries) compact();

  return {
    /** True when this (chatId, messageId) was already recorded. */
    has(chatId, messageId) {
      return seen.has(receiptKey(chatId, messageId));
    },

    /**
     * Record a receipt. Returns true when the message is new (caller should
     * process it) and false when it is a duplicate (caller should drop it).
     * The durable append happens before the in-memory add.
     */
    record(chatId, messageId) {
      const key = receiptKey(chatId, messageId);
      if (seen.has(key)) return false;
      try {
        appendFileSync(filePath, key + '\n');
      } catch (err) {
        // Fail open on write errors: forwarding a rare duplicate is cheaper
        // than silently dropping real messages when the disk hiccups.
        logger.warn?.(`[bridge] ingress receipt write failed: ${err.message}`);
      }
      seen.add(key);
      if (seen.size > maxEntries) compact();
      return true;
    },

    /** Number of receipts currently tracked (for tests/diagnostics). */
    size() {
      return seen.size;
    },
  };
}
