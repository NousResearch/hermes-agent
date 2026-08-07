import crypto from "node:crypto";

const HANDLE_RE = /^[a-f0-9]{48}$/;
const DELIVERY_RE = /^[a-f0-9]{48}$/;
const LEASE_RE = /^[a-f0-9]{48}$/;

export class AttachmentHandleError extends Error {
  constructor(code, message) {
    super(message);
    this.code = code;
  }
}

export class AttachmentHandleStore {
  constructor({
    maxItemBytes,
    maxTotalBytes,
    maxCount,
    ttlMs,
    leaseTtlMs = ttlMs,
    now = Date.now,
    randomBytes = crypto.randomBytes,
    setTimer = setTimeout,
    clearTimer = clearTimeout,
  }) {
    for (const [name, value] of Object.entries({
      maxItemBytes,
      maxTotalBytes,
      maxCount,
      ttlMs,
      leaseTtlMs,
    })) {
      if (!Number.isSafeInteger(value) || value <= 0) {
        throw new TypeError(`${name} must be a positive safe integer`);
      }
    }
    if (maxItemBytes > maxTotalBytes) {
      throw new TypeError("maxItemBytes cannot exceed maxTotalBytes");
    }
    this.maxItemBytes = maxItemBytes;
    this.maxTotalBytes = maxTotalBytes;
    this.maxCount = maxCount;
    this.ttlMs = ttlMs;
    this.leaseTtlMs = leaseTtlMs;
    this._now = now;
    this._randomBytes = randomBytes;
    this._setTimer = setTimer;
    this._clearTimer = clearTimer;
    this._entries = new Map();
    this._consumed = new Map();
    this._totalBytes = 0;
    this._expiryTimer = null;
  }

  _wipe(entry) {
    if (entry.wiped) return;
    entry.wiped = true;
    entry.bytes.fill(0);
    this._totalBytes -= entry.bytes.length;
  }

  _release(handle, entry, { abort = false } = {}) {
    if (this._entries.get(handle) !== entry) return false;
    this._entries.delete(handle);
    if (abort && !entry.aborted) {
      entry.aborted = true;
      this._abortLease(entry);
      try {
        entry.onExpire?.();
      } catch {
        // Expiry and shutdown remain authoritative when socket teardown fails.
      }
    }
    this._wipe(entry);
    return true;
  }

  _scheduleExpiry() {
    if (this._expiryTimer !== null) {
      this._clearTimer(this._expiryTimer);
      this._expiryTimer = null;
    }
    let earliest = Infinity;
    for (const entry of this._entries.values()) {
      earliest = Math.min(earliest, entry.expiresAt);
      if (entry.lease) earliest = Math.min(earliest, entry.lease.expiresAt);
    }
    for (const receipt of this._consumed.values()) {
      earliest = Math.min(earliest, receipt.expiresAt);
    }
    if (earliest === Infinity) return;
    this._expiryTimer = this._setTimer(
      () => {
        this._expiryTimer = null;
        this.purgeExpired();
      },
      Math.max(1, earliest - this._now())
    );
    this._expiryTimer?.unref?.();
  }

  purgeExpired() {
    const now = this._now();
    for (const [handle, entry] of this._entries) {
      if (entry.expiresAt <= now) {
        this._release(handle, entry, { abort: true });
      } else if (entry.lease?.expiresAt <= now) {
        this._abortLease(entry);
      }
    }
    for (const [handle, receipt] of this._consumed) {
      if (receipt.expiresAt <= now) this._consumed.delete(handle);
    }
    this._scheduleExpiry();
  }

  assertCapacityFor(size) {
    if (!Number.isSafeInteger(size) || size <= 0) {
      throw new AttachmentHandleError(
        "size_unavailable",
        "attachment size is unavailable"
      );
    }
    if (size > this.maxItemBytes) {
      throw new AttachmentHandleError(
        "item_too_large",
        "attachment exceeds the per-item limit"
      );
    }
    this.purgeExpired();
    if (
      this._entries.size >= this.maxCount ||
      this._totalBytes + size > this.maxTotalBytes
    ) {
      throw new AttachmentHandleError(
        "capacity_exceeded",
        "attachment handle capacity is full"
      );
    }
  }

  availableBytes() {
    this.purgeExpired();
    if (this._entries.size >= this.maxCount) return 0;
    return Math.min(
      this.maxItemBytes,
      this.maxTotalBytes - this._totalBytes
    );
  }

  put(bytes, { mimeType = null } = {}) {
    if (!Buffer.isBuffer(bytes)) {
      throw new TypeError("attachment bytes must be a Buffer");
    }
    this.assertCapacityFor(bytes.length);
    let handle;
    do {
      handle = this._randomBytes(24).toString("hex");
    } while (this._entries.has(handle));
    const owned = Buffer.from(bytes);
    this._entries.set(handle, {
      bytes: owned,
      mimeType:
        typeof mimeType === "string" && mimeType.length <= 255
          ? mimeType
          : null,
      expiresAt: this._now() + this.ttlMs,
      state: "queued",
      deliveryId: null,
      lease: null,
      wiped: false,
      aborted: false,
      onExpire: null,
    });
    this._totalBytes += owned.length;
    this._scheduleExpiry();
    return { handle };
  }

  _validateBinding(handle, deliveryId, leaseId = null) {
    if (typeof handle !== "string" || !HANDLE_RE.test(handle)) {
      throw new AttachmentHandleError(
        "not_found",
        "attachment handle not found"
      );
    }
    if (typeof deliveryId !== "string" || !DELIVERY_RE.test(deliveryId)) {
      throw new AttachmentHandleError("invalid_binding", "invalid delivery binding");
    }
    if (leaseId !== null && (typeof leaseId !== "string" || !LEASE_RE.test(leaseId))) {
      throw new AttachmentHandleError("invalid_binding", "invalid lease binding");
    }
  }

  _abortLease(entry) {
    if (!entry.lease) return;
    for (const abort of entry.lease.aborters) {
      try {
        abort();
      } catch {
        // Lease expiry remains authoritative when socket teardown fails.
      }
    }
    entry.lease = null;
  }

  bindHandles(handles, deliveryId) {
    if (typeof deliveryId !== "string" || !DELIVERY_RE.test(deliveryId)) {
      throw new AttachmentHandleError("invalid_binding", "invalid delivery binding");
    }
    if (!Array.isArray(handles) || handles.length === 0 || handles.length > this.maxCount) {
      throw new AttachmentHandleError("binding_failed", "attachment binding failed");
    }
    this.purgeExpired();
    const unique = new Set(handles);
    if (unique.size !== handles.length) {
      throw new AttachmentHandleError("binding_failed", "attachment binding failed");
    }
    const entries = [];
    for (const handle of handles) {
      if (typeof handle !== "string" || !HANDLE_RE.test(handle)) {
        throw new AttachmentHandleError("binding_failed", "attachment binding failed");
      }
      const entry = this._entries.get(handle);
      if (!entry || entry.deliveryId !== null) {
        throw new AttachmentHandleError("binding_failed", "attachment binding failed");
      }
      entries.push(entry);
    }
    for (const entry of entries) entry.deliveryId = deliveryId;
  }

  bindEvent(event, deliveryId) {
    const handles = [];
    const stack = [event?.content];
    while (stack.length > 0 && handles.length <= this.maxCount) {
      const content = stack.pop();
      if (!content || typeof content !== "object") continue;
      if (content.type === "attachment" || content.type === "voice") {
        if (HANDLE_RE.test(String(content.handle || ""))) handles.push(content.handle);
      } else if (content.type === "group") {
        for (const item of Array.isArray(content.items) ? content.items : []) {
          stack.push(item?.content);
        }
      }
    }
    this.bindHandles(handles, deliveryId);
  }

  lease(handle, deliveryId, { onExpire = null } = {}) {
    this._validateBinding(handle, deliveryId);
    this.purgeExpired();
    const entry = this._entries.get(handle);
    if (!entry) {
      throw new AttachmentHandleError(
        "not_found",
        "attachment handle not found"
      );
    }
    if (entry.deliveryId !== deliveryId) {
      throw new AttachmentHandleError("binding_mismatch", "attachment delivery is not bound");
    }
    if (entry.lease && entry.lease.deliveryId !== deliveryId) {
      throw new AttachmentHandleError("binding_mismatch", "attachment lease is bound");
    }
    if (!entry.lease) {
      entry.lease = {
        deliveryId,
        leaseId: this._randomBytes(24).toString("hex"),
        expiresAt: Math.min(entry.expiresAt, this._now() + this.leaseTtlMs),
        aborters: new Set(),
      };
    }
    if (typeof onExpire === "function") entry.lease.aborters.add(onExpire);
    this._scheduleExpiry();
    return { entry, leaseId: entry.lease.leaseId };
  }

  detachLeaseResponse(handle, leaseId, aborter) {
    const entry = this._entries.get(handle);
    if (entry?.lease?.leaseId === leaseId) entry.lease.aborters.delete(aborter);
  }

  finalize(handle, deliveryId) {
    this._validateBinding(handle, deliveryId);
    this.purgeExpired();
    const receipt = this._consumed.get(handle);
    if (receipt) {
      if (receipt.deliveryId === deliveryId) {
        return "duplicate";
      }
      throw new AttachmentHandleError("not_found", "attachment handle not found");
    }
    const entry = this._entries.get(handle);
    if (!entry) throw new AttachmentHandleError("not_found", "attachment handle not found");
    if (entry.deliveryId !== deliveryId || entry.lease?.deliveryId !== deliveryId) {
      throw new AttachmentHandleError("binding_mismatch", "attachment lease binding mismatch");
    }
    while (this._consumed.size >= this.maxCount) {
      this._consumed.delete(this._consumed.keys().next().value);
    }
    this._consumed.set(handle, {
      deliveryId,
      expiresAt: this._now() + this.ttlMs,
    });
    this._release(handle, entry);
    this._scheduleExpiry();
    return "consumed";
  }

  releaseLease(handle, deliveryId, leaseId) {
    this._validateBinding(handle, deliveryId, leaseId);
    this.purgeExpired();
    const entry = this._entries.get(handle);
    if (!entry) throw new AttachmentHandleError("not_found", "attachment handle not found");
    if (
      entry.deliveryId !== deliveryId ||
      entry.lease?.deliveryId !== deliveryId ||
      entry.lease?.leaseId !== leaseId
    ) {
      throw new AttachmentHandleError("binding_mismatch", "attachment lease binding mismatch");
    }
    this._abortLease(entry);
    this._scheduleExpiry();
    return "released";
  }

  stats() {
    this.purgeExpired();
    return { count: this._entries.size, totalBytes: this._totalBytes };
  }

  close() {
    if (this._expiryTimer !== null) {
      this._clearTimer(this._expiryTimer);
      this._expiryTimer = null;
    }
    for (const [handle, entry] of this._entries) {
      this._release(handle, entry, { abort: true });
    }
    this._consumed.clear();
  }
}

export async function normalizeInboundBinaryContent(
  content,
  store,
  logError = (message) => console.error(message)
) {
  const meta = {
    type: content.type,
    id: content.id ?? null,
    name: content.name ?? null,
    mimeType: content.mimeType ?? null,
    size: typeof content.size === "number" ? content.size : null,
  };
  if (content.type === "voice" && typeof content.duration === "number") {
    meta.duration = content.duration;
  }
  try {
    if (meta.size !== null) store.assertCapacityFor(meta.size);
    const maxBytes = store.availableBytes();
    if (maxBytes <= 0) {
      throw new AttachmentHandleError(
        "capacity_exceeded",
        "attachment handle capacity is full"
      );
    }
    const bytes = await readContentBytes(content, maxBytes);
    const { handle } = store.put(bytes, { mimeType: meta.mimeType });
    meta.size = bytes.length;
    meta.handle = handle;
  } catch (error) {
    const code =
      error instanceof AttachmentHandleError ? error.code : "read_failed";
    logError(`photon-sidecar: attachment bytes unavailable (${code})`);
  }
  return meta;
}

async function readContentBytes(content, maxBytes) {
  if (typeof content.stream === "function") {
    const stream = await content.stream();
    if (!stream || typeof stream.getReader !== "function") {
      throw new AttachmentHandleError(
        "stream_unavailable",
        "attachment stream is unavailable"
      );
    }
    const reader = stream.getReader();
    const chunks = [];
    let total = 0;
    try {
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        const chunk = Buffer.from(value);
        total += chunk.length;
        if (total > maxBytes) {
          try {
            await reader.cancel();
          } catch {
            // The hard cap is authoritative even if provider cancellation
            // itself reports a transport failure.
          }
          throw new AttachmentHandleError(
            "item_too_large",
            "attachment exceeds the available memory limit"
          );
        }
        chunks.push(chunk);
      }
    } finally {
      reader.releaseLock?.();
    }
    return Buffer.concat(chunks, total);
  }
  if (
    typeof content.read !== "function" ||
    !Number.isSafeInteger(content.size) ||
    content.size <= 0
  ) {
    throw new AttachmentHandleError(
      "size_unavailable",
      "attachment has no bounded byte source"
    );
  }
  const bytes = Buffer.from(await content.read());
  if (bytes.length > maxBytes) {
    throw new AttachmentHandleError(
      "item_too_large",
      "attachment exceeds the available memory limit"
    );
  }
  return bytes;
}

export function parseAttachmentHandlePath(path) {
  if (typeof path !== "string") return null;
  const match = path.match(/^\/attachment\/([a-f0-9]{48})$/);
  return match ? match[1] : null;
}

export function parseAttachmentActionPath(path) {
  if (typeof path !== "string") return null;
  const match = path.match(/^\/attachment\/([a-f0-9]{48})\/(lease|consume|release)$/);
  return match ? { handle: match[1], action: match[2] } : null;
}

function safeMimeType(value) {
  if (
    typeof value === "string" &&
    /^[A-Za-z0-9!#$&^_.+-]+\/[A-Za-z0-9!#$&^_.+-]+$/.test(value)
  ) {
    return value;
  }
  return "application/octet-stream";
}

export function serveAttachmentLease(path, body, res, store) {
  const parsed = parseAttachmentActionPath(path);
  if (parsed?.action !== "lease") return false;
  const deliveryId = body?.deliveryId;
  let leased;
  const abort = () => res.destroy?.();
  try {
    if (!body || Object.keys(body).length !== 1) throw new AttachmentHandleError("invalid_binding", "invalid delivery binding");
    leased = store.lease(parsed.handle, deliveryId, { onExpire: abort });
  } catch (error) {
    if (!(error instanceof AttachmentHandleError)) throw error;
    res.statusCode = error.code === "invalid_binding" ? 400 : error.code === "binding_mismatch" ? 409 : 404;
    res.setHeader("Content-Type", "application/json");
    res.setHeader("Cache-Control", "no-store");
    res.end(JSON.stringify({ ok: false, error: "attachment handle not found" }));
    return true;
  }
  const { entry, leaseId } = leased;
  res.statusCode = 200;
  res.setHeader("Content-Type", safeMimeType(entry.mimeType));
  res.setHeader("Content-Length", String(entry.bytes.length));
  res.setHeader("Cache-Control", "no-store");
  res.setHeader("X-Content-Type-Options", "nosniff");
  res.setHeader("X-Hermes-Attachment-Lease-Id", leaseId);
  const detach = () => store.detachLeaseResponse(parsed.handle, leaseId, abort);
  res.once?.("close", detach);
  res.once?.("error", detach);
  try {
    res.end(entry.bytes, detach);
  } catch (error) {
    detach();
    throw error;
  }
  return true;
}

export function mutateAttachmentLease(path, body, res, store) {
  const parsed = parseAttachmentActionPath(path);
  if (!parsed || parsed.action === "lease") return false;
  try {
    const expectedKeys = parsed.action === "consume" ? "deliveryId" : "deliveryId,leaseId";
    if (!body || Object.keys(body).sort().join(",") !== expectedKeys) {
      throw new AttachmentHandleError("invalid_binding", "invalid lease binding");
    }
    const status = parsed.action === "consume"
      ? store.finalize(parsed.handle, body.deliveryId)
      : store.releaseLease(parsed.handle, body.deliveryId, body.leaseId);
    res.statusCode = 200;
    res.setHeader("Content-Type", "application/json");
    res.setHeader("Cache-Control", "no-store");
    res.end(JSON.stringify({ ok: true, status }));
  } catch (error) {
    if (!(error instanceof AttachmentHandleError)) throw error;
    res.statusCode = error.code === "invalid_binding" ? 400 : error.code === "binding_mismatch" ? 409 : 404;
    res.setHeader("Content-Type", "application/json");
    res.setHeader("Cache-Control", "no-store");
    res.end(JSON.stringify({ ok: false, error: "attachment handle not found" }));
  }
  return true;
}
