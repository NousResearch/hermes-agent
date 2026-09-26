/**
 * Inbound message redelivery guard for the WhatsApp bridge.
 *
 * When WhatsApp never receives an ack for a message (e.g. the fromMe echo
 * copy of a thread whose encryption session is missing — "No matching
 * sessions found"), the server re-delivers the SAME message (same key.id)
 * roughly every 10 minutes forever.  Without a guard, every redelivery is
 * forwarded as a fresh inbound and the gateway answers the same message
 * repeatedly ("obsessed agent" / answer loop).
 *
 * This module tracks forwarded inbound ids and persists them to disk
 * (atomic tmp+rename) so a gateway/bridge restart cannot re-trigger the
 * loop either.  The id set is bounded (FIFO, see outbound_ids.js), so the
 * file stays small under sustained traffic.
 */
import { mkdirSync, readFileSync, writeFileSync, renameSync } from 'fs';
import { dirname } from 'path';

import { createOutboundIdTracker } from './outbound_ids.js';

export function createInboundDedupe({ file, maxSize = 1024 } = {}) {
  if (!file) {
    throw new RangeError('createInboundDedupe: file is required');
  }

  const ids = createOutboundIdTracker(maxSize);
  try {
    const saved = JSON.parse(readFileSync(file, 'utf8'));
    if (Array.isArray(saved)) {
      for (const id of saved) ids.remember(id);
    }
  } catch {
    // Missing or corrupt file: start clean, the guard only needs the ids
    // of messages forwarded since the last successful write.
  }

  let saveTimer = null;

  function flush() {
    if (saveTimer) {
      clearTimeout(saveTimer);
      saveTimer = null;
    }
    try {
      mkdirSync(dirname(file), { recursive: true });
      const tmp = `${file}.tmp`;
      writeFileSync(tmp, JSON.stringify(ids.snapshot()));
      renameSync(tmp, file);
      return true;
    } catch (err) {
      console.warn('[whatsapp-bridge] failed to persist inbound dedupe:', err.message);
      return false;
    }
  }

  // Debounced persist: redelivery bursts (and normal message volume) would
  // otherwise write the whole set on every single message.
  function persist() {
    if (saveTimer) clearTimeout(saveTimer);
    saveTimer = setTimeout(() => {
      saveTimer = null;
      flush();
    }, 1500);
  }

  /**
   * @param {string} id message key.id
   * @returns {boolean} true when the id was ALREADY forwarded (redelivery)
   */
  function checkAndRemember(id) {
    if (ids.has(id)) return true;
    ids.remember(id);
    persist();
    return false;
  }

  function size() {
    return ids.size();
  }

  return { checkAndRemember, flush, size };
}