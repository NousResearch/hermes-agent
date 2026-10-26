import { proto, generateMessageIDV2 } from '@whiskeysockets/baileys';
import { openSync, writeFileSync, fsyncSync, closeSync } from 'node:fs';

const Status = proto.WebMessageInfo.Status;
const accepted = new Set([Status.SERVER_ACK, Status.DELIVERY_ACK, Status.READ, Status.PLAYED]);
const statusName = status => Object.entries(Status).find(([, value]) => value === status)?.[0] || 'UNKNOWN';

// No body or exception text: errors can contain message content or credentials.
export function appendAckReceipt(file, receipt) {
  const fd = openSync(file, 'a', 0o600);
  try {
    writeFileSync(fd, JSON.stringify(receipt) + '\n');
    fsyncSync(fd);
  } finally {
    closeSync(fd);
  }
}

function bounded(operation, ms, code) {
  let timer;
  return Promise.race([
    Promise.resolve().then(operation),
    new Promise((_, reject) => { timer = setTimeout(() => reject(new Error(code)), ms); }),
  ]).finally(() => clearTimeout(timer));
}

export function createAckSender({ getSocket, send, record,
  metadataTimeoutMs = 2000, sendTimeoutMs = 5000, ackTimeoutMs = 5000,
}) {
  // Register before sendMessage so a fast server ack cannot outrun its local return.
  // Only active sends are retained; receipt history is durable, not an unbounded Map.
  const active = new Map();
  function observe(key, update) {
    const entry = active.get(key?.id);
    if (!entry || key?.remoteJid !== entry.chatId || key?.fromMe === false) return;
    if (update?.status !== Status.ERROR && !accepted.has(update?.status)) return;
    entry.status = update.status;
    entry.wake?.();
  }

  async function sendAcknowledged(chatId, payload, options = {}) {
    const socket = getSocket();
    const messageId = generateMessageIDV2(socket.user?.id);
    const entry = { chatId, status: Status.PENDING, wake: null };
    active.set(messageId, entry);
    let sent;
    try {
      const attempts = chatId.endsWith('@g.us') ? 2 : 1;
      for (let attempt = 1; attempt <= attempts; attempt++) {
        let error = null;
        try {
          if (chatId.endsWith('@g.us')) {
            await bounded(() => socket.groupMetadata(chatId), metadataTimeoutMs, 'METADATA_TIMEOUT');
          }
          // Reuse the protocol id on retry to avoid creating a second logical message.
          // A late ack during metadata refresh also avoids a needless second send.
          if (!sent || !accepted.has(entry.status)) {
            entry.status = Status.PENDING;
            sent = await send(chatId, payload, { ...options, messageId }, sendTimeoutMs);
          }
          if (sent?.key?.id !== messageId || sent?.key?.remoteJid !== chatId) {
            throw new Error('INVALID_MESSAGE_KEY');
          }
          if (!accepted.has(entry.status) && entry.status !== Status.ERROR) {
            await bounded(() => new Promise(resolve => {
              entry.wake = resolve;
              if (accepted.has(entry.status) || entry.status === Status.ERROR) resolve();
            }), ackTimeoutMs, 'ACK_TIMEOUT');
          }
          if (entry.status === Status.ERROR) throw new Error('SERVER_ERROR');
          if (!accepted.has(entry.status)) throw new Error('ACK_TIMEOUT');
        } catch (err) {
          error = ['METADATA_TIMEOUT', 'INVALID_MESSAGE_KEY', 'ACK_TIMEOUT', 'SERVER_ERROR'].includes(err.message)
            ? err.message : 'SEND_FAILED';
        }
        entry.wake = null;
        const receipt = {
          timestamp: new Date().toISOString(), message_id: messageId, chat_id: chatId,
          status: statusName(entry.status), status_code: entry.status, error, attempt,
        };
        // Audit failure is fail-closed and must never trigger a duplicate send.
        try { record(receipt); } catch {
          throw Object.assign(new Error('ACK_AUDIT_FAILED'), { status: receipt.status, messageId });
        }
        if (!error) return { ...sent, ackStatus: receipt.status };
        if (attempt === attempts) {
          throw Object.assign(new Error(error), { status: receipt.status, messageId });
        }
      }
    } finally {
      active.delete(messageId);
    }
  }
  return { observe, send: sendAcknowledged };
}
