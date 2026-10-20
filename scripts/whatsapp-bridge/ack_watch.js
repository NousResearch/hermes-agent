/**
 * Server-ack watch for outbound WhatsApp messages.
 *
 * sock.sendMessage() resolving only means Baileys wrote to the socket. On
 * 2026-10-17 a reconnected socket swallowed every send for ~36h while
 * /health said "connected" and every layer above logged success. WhatsApp's
 * server acks each message it accepts (`CB:ack,class:message`); a delivery or
 * read receipt implies the same. A send that gets neither within `staleMs`
 * makes the bridge report itself unhealthy, so the gateway restarts it.
 *
 * `sends(chatId, sinceMs)` answers, per chat, how much of what the bridge sent
 * since a moment WhatsApp acked, so the capture hook records a reply as
 * accepted only on WhatsApp's word (Go2Stone/freightman#2463).
 */

/** Same shape the gateway posts to /send (to_whatsapp_jid): bare digits get the phone domain, devices are dropped. */
export function normalizeJid(value) {
  const raw = String(value || '').trim();
  if (!raw) return '';
  const at = raw.indexOf('@');
  const user = at === -1 ? raw : raw.slice(0, at);
  const domain = at === -1 ? 's.whatsapp.net' : raw.slice(at + 1);
  return `${user.split(':', 1)[0]}@${domain}`;
}

export function createAckWatch({ staleMs = 120_000, now = () => Date.now(), keep = 1000 } = {}) {
  const pending = new Map();
  const log = new Map();
  const stats = { tracked: 0, acked: 0 };

  function acked(id) {
    if (pending.delete(id)) stats.acked++;
    const entry = log.get(id);
    if (entry) entry.acked = true;
  }

  return {
    /** Start watching a new socket; sends from the old one cannot be acked any more. */
    attach(socket) {
      pending.clear();
      socket.ws.on('CB:ack,class:message', (node) => acked(node?.attrs?.id));
      socket.ev.on('messages.update', (updates) => {
        for (const { key, update } of updates || []) {
          if (key?.fromMe && typeof update?.status === 'number' && update.status >= 2) acked(key.id);
        }
      });
    },
    sent(id, chatId) {
      if (!id) return;
      pending.set(id, now());
      log.set(id, { chatId: normalizeJid(chatId), at: now(), acked: false });
      while (log.size > keep) log.delete(log.keys().next().value);
      stats.tracked++;
    },
    /** How many sends to `chatId` at or after `sinceMs` the bridge made, and how many WhatsApp acked. */
    sends(chatId, sinceMs = 0) {
      const target = normalizeJid(chatId);
      let sent = 0;
      let ackedCount = 0;
      for (const entry of log.values()) {
        if (entry.chatId !== target || entry.at < sinceMs) continue;
        sent++;
        if (entry.acked) ackedCount++;
      }
      return { sent, acked: ackedCount };
    },
    health() {
      let oldest = 0;
      for (const at of pending.values()) oldest = Math.max(oldest, now() - at);
      return {
        stale: oldest > staleMs,
        unackedSends: pending.size,
        oldestUnackedSec: Math.round(oldest / 1000),
        sendsTracked: stats.tracked,
        sendsAcked: stats.acked,
      };
    },
  };
}

/** /health status: anything but "connected" makes the gateway recover the bridge. */
export function bridgeStatus(connectionState, { stale }) {
  return connectionState === 'connected' && stale ? 'sends_unacked' : connectionState;
}
