/**
 * Server-ack watch for outbound WhatsApp messages.
 *
 * sock.sendMessage() resolving only means Baileys wrote to the socket. On
 * 2026-10-17 a reconnected socket swallowed every send for ~36h while
 * /health said "connected" and every layer above logged success. WhatsApp's
 * server acks each message it accepts (`CB:ack,class:message`); a delivery or
 * read receipt implies the same. A send that gets neither within `staleMs`
 * makes the bridge report itself unhealthy, so the gateway restarts it.
 */
export function createAckWatch({ staleMs = 120_000, now = () => Date.now() } = {}) {
  const pending = new Map();
  const stats = { tracked: 0, acked: 0 };

  function acked(id) {
    if (pending.delete(id)) stats.acked++;
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
    sent(id) {
      if (!id) return;
      pending.set(id, now());
      stats.tracked++;
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
