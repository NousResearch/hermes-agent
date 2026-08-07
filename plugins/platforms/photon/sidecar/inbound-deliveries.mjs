import crypto from "node:crypto";
import { once } from "node:events";

const DELIVERY_ID_RE = /^[a-f0-9]{48}$/;

export function eventHasAttachmentHandle(event) {
  function hasHandle(content) {
    if (!content || typeof content !== "object") return false;
    if (content.type === "attachment" || content.type === "voice") {
      return DELIVERY_ID_RE.test(String(content.handle || ""));
    }
    if (content.type === "group") {
      return (Array.isArray(content.items) ? content.items : []).some((item) =>
        hasHandle(item?.content)
      );
    }
    return false;
  }
  return hasHandle(event?.content);
}

export function parseDeliveryAck(body) {
  if (!body || typeof body !== "object" || Array.isArray(body)) {
    throw new Error("invalid_delivery_ack");
  }
  const keys = Object.keys(body);
  if (keys.length !== 1 || keys[0] !== "deliveryId" ||
      !DELIVERY_ID_RE.test(String(body.deliveryId || ""))) {
    throw new Error("invalid_delivery_ack");
  }
  return body.deliveryId;
}

export async function deliverPendingEntry(entry, transport) {
  for (;;) {
    await transport.waitForConsumer();
    const { res, version } = transport.currentConsumer();
    if (!res) continue;
    try {
      const flushed = res.write(entry.line + "\n");
      if (!flushed) {
        const drain = once(res, "drain").then(() => "drained");
        const changed = transport.waitForConsumerChange(version);
        const outcome = await Promise.race([entry.settled, changed.promise, drain]);
        changed.cancel();
        if (outcome === "acked" || outcome === "expired" || outcome === "closed") {
          return outcome;
        }
        if (outcome === "consumer_changed") continue;
      }
      const changed = transport.waitForConsumerChange(version);
      const outcome = await Promise.race([entry.settled, changed.promise]);
      changed.cancel();
      if (outcome === "consumer_changed") continue;
      return outcome;
    } catch {
      transport.clearConsumer(res);
    }
  }
}

export class InboundDeliveryQueue {
  constructor({
    maxBytes = 2 * 1024 * 1024,
    ttlMs = 5 * 60 * 1000,
    recentAckMax = 64,
    randomBytes = crypto.randomBytes,
    setTimer = setTimeout,
    clearTimer = clearTimeout,
  } = {}) {
    this.maxBytes = maxBytes;
    this.ttlMs = ttlMs;
    this.recentAckMax = recentAckMax;
    this.randomBytes = randomBytes;
    this.setTimer = setTimer;
    this.clearTimer = clearTimer;
    this.pending = null;
    this.totalBytes = 0;
    this.recentAcks = new Map();
  }

  begin(event, { bindDelivery = null } = {}) {
    if (this.pending) throw new Error("delivery_queue_full");
    const deliveryId = this.randomBytes(24).toString("hex");
    const line = JSON.stringify({ ...event, deliveryId });
    const bytes = Buffer.byteLength(line);
    if (bytes > this.maxBytes) throw new Error("delivery_too_large");
    if (typeof bindDelivery === "function") bindDelivery(deliveryId);
    let settle;
    const settled = new Promise((resolve) => { settle = resolve; });
    const entry = { deliveryId, line, bytes, settled, settle, timer: null };
    entry.timer = this.setTimer(() => this._release(entry, "expired"), this.ttlMs);
    if (entry.timer && typeof entry.timer.unref === "function") entry.timer.unref();
    this.pending = entry;
    this.totalBytes = bytes;
    return entry;
  }

  _rememberAck(deliveryId) {
    this.recentAcks.delete(deliveryId);
    this.recentAcks.set(deliveryId, true);
    while (this.recentAcks.size > this.recentAckMax) {
      this.recentAcks.delete(this.recentAcks.keys().next().value);
    }
  }

  _release(entry, outcome) {
    if (this.pending !== entry) return false;
    this.clearTimer(entry.timer);
    this.pending = null;
    this.totalBytes = 0;
    if (outcome === "acked") this._rememberAck(entry.deliveryId);
    entry.settle(outcome);
    return true;
  }

  ack(deliveryId) {
    if (this.pending?.deliveryId === deliveryId) {
      this._release(this.pending, "acked");
      return "acked";
    }
    return this.recentAcks.has(deliveryId) ? "duplicate" : "not_found";
  }

  stats() {
    return { count: this.pending ? 1 : 0, totalBytes: this.totalBytes };
  }

  close() {
    if (this.pending) this._release(this.pending, "closed");
    this.recentAcks.clear();
  }
}
