/**
 * Passive-capture helpers for the WhatsApp bridge.
 *
 * Passive capture turns the bridge into an ingress-only archive: every
 * inbound message from another person is appended to a private NDJSON file
 * and is never handed to the Python adapter (so no agent turn, no typing
 * bubble, no read receipt, no reply).  Kept out of bridge.js so the decision
 * and the file handling can be unit-tested without Baileys or Express.
 */

import { appendFileSync, chmodSync, closeSync, fstatSync, mkdirSync, openSync, readSync } from 'fs';
import path from 'path';

/**
 * True when the message must end in the passive archive instead of the agent
 * queue.  Own (fromMe) messages are never captured: they are either echoes of
 * our own sends or owner-typed messages, and neither is new inbound traffic.
 */
export function isPassiveTerminal({ enabled, fromMe }) {
  return Boolean(enabled) && !fromMe;
}

/**
 * Build the archived record, or null when the event carries nothing worth
 * reporting (reactions, protocol/edit/delete stubs, empty payloads).
 */
export function buildPassiveRecord(event, now = new Date()) {
  const body = String(event?.body || '');
  const hasMedia = Boolean(event?.hasMedia);
  if (!body.trim() && !hasMedia) return null;
  return {
    capturedAt: now.toISOString(),
    direction: 'inbound',
    chatId: event.chatId,
    senderId: event.senderId,
    senderName: event.senderName || '',
    isGroup: Boolean(event.isGroup),
    messageId: event.messageId || '',
    timestamp: event.timestamp || null,
    body,
    hasMedia,
    mediaType: event.mediaType || '',
    mediaUrls: event.mediaUrls || [],
  };
}

function endsWithoutNewline(file) {
  let fd;
  try {
    fd = openSync(file, 'r');
    const { size } = fstatSync(fd);
    if (size === 0) return false;
    const buf = Buffer.alloc(1);
    readSync(fd, buf, 0, 1, size - 1);
    return buf[0] !== 0x0a;
  } catch (err) {
    if (err?.code === 'ENOENT') return false;
    throw err;
  } finally {
    if (fd !== undefined) closeSync(fd);
  }
}

/**
 * Append one NDJSON line.  A torn final line left by a crash is terminated
 * first, so it cannot swallow the next record.  The file is (re)forced to
 * 0600 because appendFileSync's mode only applies when it creates the file.
 */
export function appendPassiveRecord(file, record) {
  mkdirSync(path.dirname(file), { recursive: true, mode: 0o700 });
  const prefix = endsWithoutNewline(file) ? '\n' : '';
  appendFileSync(file, `${prefix}${JSON.stringify(record)}\n`, { encoding: 'utf8', mode: 0o600 });
  chmodSync(file, 0o600);
}

function bareUser(jid) {
  return String(jid || '').replace(/:.*@/, '@').replace(/@.*/, '');
}

/**
 * True when chatId is the account's own "Message yourself" chat (phone or LID
 * form).  Passive mode keeps exactly this chat as the owner's command channel.
 */
export function isOwnSelfChat(chatId, user) {
  const chat = bareUser(chatId);
  if (!chat || String(chatId || '').endsWith('@g.us')) return false;
  const myNumber = bareUser(user?.id);
  const myLid = bareUser(user?.lid);
  return Boolean((myNumber && chat === myNumber) || (myLid && chat === myLid));
}
