import assert from "node:assert/strict";
import test from "node:test";

import {
  deliverPendingEntry,
  InboundDeliveryQueue,
  eventHasAttachmentHandle,
  parseDeliveryAck,
} from "./inbound-deliveries.mjs";

function timers() {
  const slots = [];
  return {
    slots,
    setTimer(fn) {
      slots.push(fn);
      return fn;
    },
    clearTimer() {},
  };
}

test("attachment delivery is content-bound, opaque, and charged", () => {
  const t = timers();
  const queue = new InboundDeliveryQueue({
    maxBytes: 4096,
    ttlMs: 600000,
    randomBytes: () => Buffer.alloc(24, 0xab),
    setTimer: t.setTimer,
    clearTimer: t.clearTimer,
  });
  const event = { messageId: "m1", content: { type: "attachment", handle: "a".repeat(48) } };
  const entry = queue.begin(event);

  assert.equal(entry.deliveryId, "ab".repeat(24));
  assert.equal(event.deliveryId, undefined);
  assert.equal(JSON.parse(entry.line).deliveryId, entry.deliveryId);
  assert.deepEqual(queue.stats(), { count: 1, totalBytes: Buffer.byteLength(entry.line) });
});

test("delivery binding runs before an event becomes pending and fails atomically", () => {
  const t = timers();
  const queue = new InboundDeliveryQueue({
    maxBytes: 4096,
    ttlMs: 600000,
    randomBytes: () => Buffer.alloc(24, 0xcd),
    setTimer: t.setTimer,
    clearTimer: t.clearTimer,
  });
  let bound = null;
  const entry = queue.begin(
    { content: { type: "attachment", handle: "a".repeat(48) } },
    { bindDelivery: (deliveryId) => { bound = deliveryId; } }
  );
  assert.equal(bound, entry.deliveryId);
  queue.ack(entry.deliveryId);

  assert.throws(
    () => queue.begin(
      { content: { type: "attachment", handle: "b".repeat(48) } },
      { bindDelivery: () => { throw new Error("binding_failed"); } }
    ),
    /binding_failed/
  );
  assert.deepEqual(queue.stats(), { count: 0, totalBytes: 0 });
});

test("one pending event and byte limit are hard bounds", () => {
  const t = timers();
  const queue = new InboundDeliveryQueue({
    maxBytes: 128,
    ttlMs: 600000,
    setTimer: t.setTimer,
    clearTimer: t.clearTimer,
  });
  assert.throws(() => queue.begin({ content: { text: "x".repeat(500) } }), /delivery_too_large/);
  queue.begin({ content: { text: "ok" } });
  assert.throws(() => queue.begin({ content: { text: "second" } }), /delivery_queue_full/);
});

test("ack is idempotent and releases capacity", async () => {
  const t = timers();
  const queue = new InboundDeliveryQueue({ ttlMs: 600000, setTimer: t.setTimer, clearTimer: t.clearTimer });
  const entry = queue.begin({ content: { text: "ok" } });
  assert.equal(queue.ack(entry.deliveryId), "acked");
  assert.equal(await entry.settled, "acked");
  assert.equal(queue.ack(entry.deliveryId), "duplicate");
  assert.equal(queue.ack("f".repeat(48)), "not_found");
  assert.deepEqual(queue.stats(), { count: 0, totalBytes: 0 });
});

test("ttl expiry releases capacity without retaining content", async () => {
  const t = timers();
  const queue = new InboundDeliveryQueue({ ttlMs: 600000, setTimer: t.setTimer, clearTimer: t.clearTimer });
  const entry = queue.begin({ content: { text: "private-value" } });
  t.slots[0]();
  assert.equal(await entry.settled, "expired");
  assert.deepEqual(queue.stats(), { count: 0, totalBytes: 0 });
});

test("handle detection includes mixed group children", () => {
  assert.equal(eventHasAttachmentHandle({ content: { type: "text", text: "x" } }), false);
  assert.equal(eventHasAttachmentHandle({ content: { type: "group", items: [
    { content: { type: "text", text: "x" } },
    { content: { type: "voice", handle: "c".repeat(48) } },
  ] } }), true);
});

test("ack body is exact and token-shaped", () => {
  assert.equal(parseDeliveryAck({ deliveryId: "d".repeat(48) }), "d".repeat(48));
  assert.throws(() => parseDeliveryAck({ deliveryId: "d".repeat(48), extra: true }), /invalid_delivery_ack/);
  assert.throws(() => parseDeliveryAck({ deliveryId: "nope" }), /invalid_delivery_ack/);
});

test("pending delivery replays the identical line after consumer reconnect", async () => {
  const t = timers();
  const queue = new InboundDeliveryQueue({ ttlMs: 600000, setTimer: t.setTimer, clearTimer: t.clearTimer });
  const entry = queue.begin({ messageId: "ordered", content: { type: "attachment", handle: "e".repeat(48) } });
  const first = { lines: [], write(line) { this.lines.push(line); return true; } };
  const second = { lines: [], write(line) { this.lines.push(line); return true; } };
  let res = first;
  let version = 1;
  let changedResolve;
  const transport = {
    async waitForConsumer() {},
    currentConsumer: () => ({ res, version }),
    waitForConsumerChange(expected) {
      if (expected !== version) {
        return { promise: Promise.resolve("consumer_changed"), cancel() {} };
      }
      let cancelled = false;
      const promise = new Promise((resolve) => {
        changedResolve = () => { if (!cancelled) resolve("consumer_changed"); };
      });
      return { promise, cancel() { cancelled = true; } };
    },
    clearConsumer() {},
  };

  const delivered = deliverPendingEntry(entry, transport);
  await new Promise((resolve) => setImmediate(resolve));
  assert.deepEqual(first.lines, [entry.line + "\n"]);
  res = second;
  version += 1;
  changedResolve();
  await new Promise((resolve) => setImmediate(resolve));
  assert.deepEqual(second.lines, [entry.line + "\n"]);
  queue.ack(entry.deliveryId);
  assert.equal(await delivered, "acked");
});
