import assert from "node:assert/strict";
import { EventEmitter } from "node:events";
import test from "node:test";

import {
  AttachmentHandleError,
  AttachmentHandleStore,
  mutateAttachmentLease,
  normalizeInboundBinaryContent,
  parseAttachmentActionPath,
  serveAttachmentLease,
} from "./attachment-handles.mjs";

const noTimer = () => ({ unref() {} });
const DELIVERY_ID = "d".repeat(48);

test("attachment bytes are represented by metadata and an opaque leased handle", async () => {
  const store = new AttachmentHandleStore({
    maxItemBytes: 32,
    maxTotalBytes: 64,
    maxCount: 2,
    ttlMs: 1_000,
    setTimer: noTimer,
  });
  const event = await normalizeInboundBinaryContent(
    {
      type: "attachment",
      id: "provider-1",
      name: "photo.png",
      mimeType: "image/png",
      size: 4,
      read: async () => Buffer.from([1, 2, 3, 4]),
    },
    store
  );

  assert.deepEqual(Object.keys(event).sort(), [
    "handle",
    "id",
    "mimeType",
    "name",
    "size",
    "type",
  ]);
  assert.match(event.handle, /^[a-f0-9]{48}$/);
  assert.equal("data" in event, false);
  assert.equal("encoding" in event, false);

  store.bindHandles([event.handle], DELIVERY_ID);
  const leased = store.lease(event.handle, DELIVERY_ID);
  assert.deepEqual([...leased.entry.bytes], [1, 2, 3, 4]);
  assert.equal(leased.entry.mimeType, "image/png");
  assert.equal(store.lease(event.handle, DELIVERY_ID).leaseId, leased.leaseId);
});

test("store enforces item, count, and aggregate byte caps", () => {
  const store = new AttachmentHandleStore({
    maxItemBytes: 4,
    maxTotalBytes: 6,
    maxCount: 2,
    ttlMs: 1_000,
    setTimer: noTimer,
  });
  store.put(Buffer.from("123"), { mimeType: "text/plain" });
  store.put(Buffer.from("456"), { mimeType: "text/plain" });
  assert.throws(
    () => store.put(Buffer.from("7"), {}),
    (error) => error.code === "capacity_exceeded"
  );
  assert.throws(
    () => store.put(Buffer.from("12345"), {}),
    (error) => error.code === "item_too_large"
  );
  assert.deepEqual(store.stats(), { count: 2, totalBytes: 6 });
});

test("expired handles are wiped and cannot be replayed", () => {
  let now = 10_000;
  const store = new AttachmentHandleStore({
    maxItemBytes: 8,
    maxTotalBytes: 8,
    maxCount: 1,
    ttlMs: 50,
    now: () => now,
    setTimer: noTimer,
  });
  const { handle } = store.put(Buffer.from("secret"), {});
  now += 51;
  store.purgeExpired();
  assert.deepEqual(store.stats(), { count: 0, totalBytes: 0 });
  assert.throws(() => store.lease(handle, DELIVERY_ID), /not found/);
});

test("expiry timer purges bytes without a later request", () => {
  let now = 20_000;
  let expire = null;
  const store = new AttachmentHandleStore({
    maxItemBytes: 8,
    maxTotalBytes: 8,
    maxCount: 1,
    ttlMs: 25,
    now: () => now,
    setTimer: (callback) => {
      expire = callback;
      return { unref() {} };
    },
    clearTimer: () => {},
  });
  store.put(Buffer.from("secret"), {});
  assert.equal(typeof expire, "function");
  now += 26;
  expire();
  assert.deepEqual(store.stats(), { count: 0, totalBytes: 0 });
});

test("read failures and over-cap reads emit metadata without a handle", async () => {
  const errors = [];
  const store = new AttachmentHandleStore({
    maxItemBytes: 4,
    maxTotalBytes: 8,
    maxCount: 2,
    ttlMs: 1_000,
    setTimer: noTimer,
  });
  const failed = await normalizeInboundBinaryContent(
    {
      type: "attachment",
      name: "failed.bin",
      size: 4,
      read: async () => {
        throw new Error("private provider failure");
      },
    },
    store,
    (message) => errors.push(message)
  );
  const oversized = await normalizeInboundBinaryContent(
    {
      type: "attachment",
      name: "large.bin",
      size: 4,
      read: async () => Buffer.from("12345"),
    },
    store,
    (message) => errors.push(message)
  );

  assert.equal("handle" in failed, false);
  assert.equal("handle" in oversized, false);
  assert.equal(JSON.stringify(errors).includes("private provider failure"), false);
  assert.deepEqual(store.stats(), { count: 0, totalBytes: 0 });
});

test("unknown-size Spectrum streams are read incrementally within the hard cap", async () => {
  const store = new AttachmentHandleStore({
    maxItemBytes: 4,
    maxTotalBytes: 4,
    maxCount: 1,
    ttlMs: 1_000,
    setTimer: noTimer,
  });
  const content = {
    type: "attachment",
    name: "stream.bin",
    stream: async () =>
      new ReadableStream({
        start(controller) {
          controller.enqueue(Uint8Array.from([1, 2]));
          controller.enqueue(Uint8Array.from([3, 4]));
          controller.close();
        },
      }),
  };

  const event = await normalizeInboundBinaryContent(content, store);
  assert.equal(event.size, 4);
  store.bindHandles([event.handle], DELIVERY_ID);
  assert.deepEqual([...store.lease(event.handle, DELIVERY_ID).entry.bytes], [1, 2, 3, 4]);
});

test("unknown-size streams are cancelled when they cross the hard cap", async () => {
  let cancelled = false;
  const store = new AttachmentHandleStore({
    maxItemBytes: 3,
    maxTotalBytes: 3,
    maxCount: 1,
    ttlMs: 1_000,
    setTimer: noTimer,
  });
  const event = await normalizeInboundBinaryContent(
    {
      type: "attachment",
      name: "large-stream.bin",
      stream: async () =>
        new ReadableStream({
          start(controller) {
            controller.enqueue(Uint8Array.from([1, 2]));
            controller.enqueue(Uint8Array.from([3, 4]));
          },
          cancel() {
            cancelled = true;
          },
        }),
    },
    store,
    () => {}
  );

  assert.equal("handle" in event, false);
  assert.equal(cancelled, true);
  assert.deepEqual(store.stats(), { count: 0, totalBytes: 0 });
});

test("attachment lease action path schema is exact", () => {
  const handle = "a".repeat(48);
  assert.deepEqual(parseAttachmentActionPath(`/attachment/${handle}/lease`), {
    handle,
    action: "lease",
  });
  for (const path of [
    `/attachment/${handle}/lease?again=1`,
    `/attachment/${handle}/extra`,
    `/attachment/${"a".repeat(47)}`,
    `/attachments/${handle}`,
    "/attachment/not-hex",
  ]) {
    assert.equal(parseAttachmentActionPath(path), null);
  }
});

test("wrong first lease cannot claim a handle before its delivery is bound", () => {
  const store = new AttachmentHandleStore({
    maxItemBytes: 16,
    maxTotalBytes: 16,
    maxCount: 1,
    ttlMs: 1_000,
    setTimer: noTimer,
  });
  const { handle } = store.put(Buffer.from("private bytes"), {});
  const wrong = "e".repeat(48);
  assert.throws(
    () => store.lease(handle, wrong),
    (error) => error.code === "binding_mismatch"
  );
  store.bindHandles([handle], DELIVERY_ID);
  assert.throws(
    () => store.lease(handle, wrong),
    (error) => error.code === "binding_mismatch"
  );
  assert.equal(store.lease(handle, DELIVERY_ID).entry.bytes.toString(), "private bytes");
});

test("mixed group handles are atomically pre-bound to one delivery", () => {
  let counter = 1;
  const store = new AttachmentHandleStore({
    maxItemBytes: 16,
    maxTotalBytes: 32,
    maxCount: 2,
    ttlMs: 1_000,
    setTimer: noTimer,
    randomBytes: () => Buffer.alloc(24, counter++),
  });
  const first = store.put(Buffer.from("first"), {}).handle;
  const second = store.put(Buffer.from("second"), {}).handle;
  store.bindEvent({
    content: {
      type: "group",
      items: [
        { content: { type: "attachment", handle: first } },
        { content: { type: "voice", handle: second } },
      ],
    },
  }, DELIVERY_ID);
  assert.equal(store.lease(first, DELIVERY_ID).entry.bytes.toString(), "first");
  assert.equal(store.lease(second, DELIVERY_ID).entry.bytes.toString(), "second");
});

test("partial multi-handle bind failure rolls back every new binding", () => {
  let counter = 1;
  const store = new AttachmentHandleStore({
    maxItemBytes: 16,
    maxTotalBytes: 32,
    maxCount: 2,
    ttlMs: 1_000,
    setTimer: noTimer,
    randomBytes: () => Buffer.alloc(24, counter++),
  });
  const first = store.put(Buffer.from("first"), {}).handle;
  const second = store.put(Buffer.from("second"), {}).handle;
  store.bindHandles([second], "e".repeat(48));
  assert.throws(
    () => store.bindHandles([first, second], DELIVERY_ID),
    (error) => error.code === "binding_failed"
  );
  store.bindHandles([first], DELIVERY_ID);
  assert.equal(store.lease(first, DELIVERY_ID).entry.bytes.toString(), "first");
  assert.equal(store.lease(second, "e".repeat(48)).entry.bytes.toString(), "second");
});

test("missing handle in a multi-bind leaves existing handles unbound", () => {
  const store = new AttachmentHandleStore({
    maxItemBytes: 16,
    maxTotalBytes: 32,
    maxCount: 2,
    ttlMs: 1_000,
    setTimer: noTimer,
  });
  const existing = store.put(Buffer.from("private bytes"), {}).handle;
  assert.throws(
    () => store.bindHandles([existing, "f".repeat(48)], DELIVERY_ID),
    (error) => error.code === "binding_failed"
  );
  store.bindHandles([existing], DELIVERY_ID);
  assert.equal(
    store.lease(existing, DELIVERY_ID).entry.bytes.toString(),
    "private bytes"
  );
});

class FakeResponse extends EventEmitter {
  constructor() {
    super();
    this.headers = {};
    this.statusCode = null;
    this.body = null;
  }

  setHeader(name, value) {
    this.headers[name] = value;
  }

  end(body, callback) {
    this.body = Buffer.isBuffer(body) ? Buffer.from(body) : String(body ?? "");
    callback?.();
  }
}

test("consumer crash after lease replays the same bytes and lease", () => {
  const store = new AttachmentHandleStore({
    maxItemBytes: 16,
    maxTotalBytes: 16,
    maxCount: 1,
    ttlMs: 1_000,
    setTimer: noTimer,
  });
  const { handle } = store.put(Buffer.from("private bytes"), {
    mimeType: "text/plain",
  });
  store.bindHandles([handle], DELIVERY_ID);
  const first = new FakeResponse();
  assert.equal(
    serveAttachmentLease(`/attachment/${handle}/lease`, { deliveryId: DELIVERY_ID }, first, store),
    true
  );
  assert.equal(first.statusCode, 200);
  assert.equal(first.headers["Content-Type"], "text/plain");
  assert.equal(first.headers["Cache-Control"], "no-store");
  assert.equal(first.body.toString(), "private bytes");

  assert.deepEqual(store.stats(), { count: 1, totalBytes: 13 });
  const replay = new FakeResponse();
  assert.equal(
    serveAttachmentLease(`/attachment/${handle}/lease`, { deliveryId: DELIVERY_ID }, replay, store),
    true
  );
  assert.equal(replay.statusCode, 200);
  assert.equal(replay.body.toString(), "private bytes");
  assert.equal(
    replay.headers["X-Hermes-Attachment-Lease-Id"],
    first.headers["X-Hermes-Attachment-Lease-Id"]
  );
});

test("expired attachment lease returns the stable content-free not-found envelope", () => {
  let now = 1_000;
  const store = new AttachmentHandleStore({
    maxItemBytes: 16,
    maxTotalBytes: 16,
    maxCount: 1,
    ttlMs: 10,
    now: () => now,
    setTimer: noTimer,
  });
  const { handle } = store.put(Buffer.from("private bytes"), {});
  store.bindHandles([handle], DELIVERY_ID);
  now += 11;
  const expired = new FakeResponse();
  serveAttachmentLease(`/attachment/${handle}/lease`, { deliveryId: DELIVERY_ID }, expired, store);
  assert.equal(expired.statusCode, 404);
  assert.deepEqual(JSON.parse(expired.body), {
    ok: false,
    error: "attachment handle not found",
  });
});

test("response faults retain charged bytes for an exact replay", () => {
  const store = new AttachmentHandleStore({
    maxItemBytes: 16,
    maxTotalBytes: 16,
    maxCount: 1,
    ttlMs: 1_000,
    setTimer: noTimer,
  });
  const { handle } = store.put(Buffer.from("private bytes"), {});
  store.bindHandles([handle], DELIVERY_ID);
  class FailingResponse extends FakeResponse {
    end(body) {
      this.reference = body;
      throw new Error("socket failed");
    }
  }
  const response = new FailingResponse();
  assert.throws(
    () => serveAttachmentLease(`/attachment/${handle}/lease`, { deliveryId: DELIVERY_ID }, response, store),
    /socket failed/
  );
  assert.equal(response.reference.toString(), "private bytes");
  assert.deepEqual(store.stats(), { count: 1, totalBytes: 13 });
  const replay = store.lease(handle, DELIVERY_ID);
  assert.equal(replay.entry.bytes.toString(), "private bytes");
});

test("stalled responses remain charged and are wiped by the transfer TTL", () => {
  let now = 30_000;
  let expire = null;
  const store = new AttachmentHandleStore({
    maxItemBytes: 16,
    maxTotalBytes: 16,
    maxCount: 1,
    ttlMs: 50,
    now: () => now,
    setTimer: (callback) => {
      expire = callback;
      return { unref() {} };
    },
    clearTimer: () => {},
  });
  const { handle } = store.put(Buffer.from("private bytes"), {});
  store.bindHandles([handle], DELIVERY_ID);
  class StalledResponse extends FakeResponse {
    end(body) {
      this.reference = body;
    }

    destroy() {
      this.destroyed = true;
      this.emit("close");
    }
  }
  const response = new StalledResponse();
  serveAttachmentLease(`/attachment/${handle}/lease`, { deliveryId: DELIVERY_ID }, response, store);

  assert.deepEqual(store.stats(), { count: 1, totalBytes: 13 });
  assert.throws(
    () => store.put(Buffer.from("x"), {}),
    (error) => error.code === "capacity_exceeded"
  );
  now += 51;
  expire();
  assert.deepEqual(store.stats(), { count: 0, totalBytes: 0 });
  assert.equal(response.reference.every((value) => value === 0), true);
  assert.equal(response.destroyed, true);

  const replay = new FakeResponse();
  serveAttachmentLease(`/attachment/${handle}/lease`, { deliveryId: DELIVERY_ID }, replay, store);
  assert.equal(replay.statusCode, 404);
});

test("only exact binding finalizes and duplicate finalize is idempotent", () => {
  const store = new AttachmentHandleStore({
    maxItemBytes: 16,
    maxTotalBytes: 16,
    maxCount: 1,
    ttlMs: 1_000,
    setTimer: noTimer,
  });
  const { handle } = store.put(Buffer.from("private bytes"), {});
  store.bindHandles([handle], DELIVERY_ID);
  const lease = store.lease(handle, DELIVERY_ID);
  assert.throws(
    () => store.finalize(handle, "e".repeat(48)),
    (error) => error.code === "binding_mismatch"
  );
  assert.deepEqual(store.stats(), { count: 1, totalBytes: 13 });
  assert.equal(store.finalize(handle, DELIVERY_ID), "consumed");
  assert.deepEqual(store.stats(), { count: 0, totalBytes: 0 });
  assert.equal(store.finalize(handle, DELIVERY_ID), "duplicate");
  assert.throws(
    () => store.finalize(handle, "e".repeat(48)),
    (error) => error.code === "not_found"
  );
});

test("consumed idempotency receipts are strictly count bounded under flood", () => {
  let counter = 1;
  const store = new AttachmentHandleStore({
    maxItemBytes: 16,
    maxTotalBytes: 32,
    maxCount: 2,
    ttlMs: 1_000,
    setTimer: noTimer,
    randomBytes: () => Buffer.alloc(24, counter++),
  });
  for (let index = 0; index < 8; index += 1) {
    const { handle } = store.put(Buffer.from(`secret-${index}`), {});
    const deliveryId = index.toString(16).padStart(48, "0");
    store.bindHandles([handle], deliveryId);
    const lease = store.lease(handle, deliveryId);
    assert.equal(store.finalize(handle, deliveryId), "consumed");
    assert.ok(store._consumed.size <= 2);
  }
  assert.equal(store._consumed.size, 2);
  assert.deepEqual(store.stats(), { count: 0, totalBytes: 0 });
});

test("release preserves bytes and permits a fresh lease for the same delivery", () => {
  const store = new AttachmentHandleStore({
    maxItemBytes: 16,
    maxTotalBytes: 16,
    maxCount: 1,
    ttlMs: 1_000,
    setTimer: noTimer,
  });
  const { handle } = store.put(Buffer.from("private bytes"), {});
  store.bindHandles([handle], DELIVERY_ID);
  const first = store.lease(handle, DELIVERY_ID);
  assert.equal(store.releaseLease(handle, DELIVERY_ID, first.leaseId), "released");
  assert.deepEqual(store.stats(), { count: 1, totalBytes: 13 });
  const retry = store.lease(handle, DELIVERY_ID);
  assert.notEqual(retry.leaseId, first.leaseId);
  assert.equal(retry.entry.bytes.toString(), "private bytes");
});

test("consume and release HTTP mutations require exact bindings", () => {
  const store = new AttachmentHandleStore({
    maxItemBytes: 16,
    maxTotalBytes: 16,
    maxCount: 1,
    ttlMs: 1_000,
    setTimer: noTimer,
  });
  const { handle } = store.put(Buffer.from("private bytes"), {});
  store.bindHandles([handle], DELIVERY_ID);
  const lease = store.lease(handle, DELIVERY_ID);
  const consumed = new FakeResponse();
  assert.equal(mutateAttachmentLease(`/attachment/${handle}/consume`, {
    deliveryId: DELIVERY_ID,
  }, consumed, store), true);
  assert.deepEqual(JSON.parse(consumed.body), { ok: true, status: "consumed" });
  const duplicate = new FakeResponse();
  mutateAttachmentLease(`/attachment/${handle}/consume`, {
    deliveryId: DELIVERY_ID,
  }, duplicate, store);
  assert.deepEqual(JSON.parse(duplicate.body), { ok: true, status: "duplicate" });
});

test("completion close error and shutdown detach responses without consuming", () => {
  for (const releaseKind of ["completion", "close", "error", "shutdown"]) {
    const store = new AttachmentHandleStore({
      maxItemBytes: 16,
      maxTotalBytes: 16,
      maxCount: 1,
      ttlMs: 1_000,
      setTimer: noTimer,
    });
    const { handle } = store.put(Buffer.from("private bytes"), {});
    store.bindHandles([handle], DELIVERY_ID);
    class ManualResponse extends FakeResponse {
      end(body, callback) {
        this.reference = body;
        this.completion = callback;
      }

      destroy() {
        this.destroyed = true;
        this.emit("close");
      }
    }
    const response = new ManualResponse();
    serveAttachmentLease(`/attachment/${handle}/lease`, { deliveryId: DELIVERY_ID }, response, store);
    if (releaseKind === "completion") response.completion();
    if (releaseKind === "close") response.emit("close");
    if (releaseKind === "error") response.emit("error", new Error("socket"));
    if (releaseKind === "shutdown") store.close();

    response.completion();
    response.emit("close");
    if (releaseKind !== "error") response.emit("error", new Error("late"));
    assert.deepEqual(
      store.stats(),
      releaseKind === "shutdown" ? { count: 0, totalBytes: 0 } : { count: 1, totalBytes: 13 }
    );
    assert.equal(
      response.reference.every((value) => value === 0),
      releaseKind === "shutdown"
    );
    assert.equal(Boolean(response.destroyed), releaseKind === "shutdown");
    store.close();
  }
});
