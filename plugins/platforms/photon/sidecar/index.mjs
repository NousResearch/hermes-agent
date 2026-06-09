// Hermes Agent — Photon Spectrum sidecar
//
// Spawned by `plugins/platforms/photon/adapter.py` to bridge messaging to
// Photon's Spectrum platform. Outbound calls require the spectrum-ts SDK
// (Photon has no public HTTP send endpoint today). Inbound is bridged here too:
// the sidecar consumes spectrum-ts' `app.messages` gRPC stream and forwards
// each message to the adapter's loopback webhook — necessary because attachment
// bytes are only reachable from this Node process (the SDK's read()/stream()
// closures are lost on JSON serialization), so the sidecar downloads them to a
// temp file and forwards the local path.
//
// Protocol:
//   - The sidecar listens on http://127.0.0.1:${PORT} (loopback only)
//   - Each request must include `X-Hermes-Sidecar-Token: ${TOKEN}`
//   - POST /healthz                     -> {"ok": true}
//   - POST /send                        -> {"ok": true, "messageId": "..."}
//       body: {"spaceId": "...", "text": "...", "effect": "confetti" | null}
//       (a URL in `text` renders a native link preview automatically)
//   - POST /send-attachment             -> {"ok": true, "messageId": "..."}
//       body: {"spaceId": "...", "path": "...", "name": "..." | null,
//              "mimeType": "..." | null, "caption": "..." | null,
//              "kind": "attachment" | "voice", "effect": "slam" | null}
//   - POST /send-attachments            -> {"ok": true, "messageIds": [...]}
//       body: {"spaceId": "...", "paths": ["...", ...],
//              "caption": "..." | null, "pacingMs": 600 | null}
//   - POST /typing                      -> {"ok": true}
//       body: {"spaceId": "...", "state": "start" | "stop"}
//   - POST /react                       -> {"ok": true}
//       body: {"spaceId": "...", "targetMessageId": "...", "reaction": "👍"}
//   - POST /shutdown                    -> {"ok": true}; then process exits
//
// On SIGINT/SIGTERM the sidecar calls `app.stop()` (3s graceful) before
// exiting. Errors are logged to stderr; Python supervises restart.
//
// Env vars (all required):
//   PHOTON_PROJECT_ID
//   PHOTON_PROJECT_SECRET
//   PHOTON_SIDECAR_PORT
//   PHOTON_SIDECAR_TOKEN
//
// Optional:
//   PHOTON_SIDECAR_BIND  (default 127.0.0.1)
//   PHOTON_API_HOST      (passed through to spectrum-ts if its config
//                         honours it)

import http from "node:http";
import crypto from "node:crypto";
import fs from "node:fs";
import fsp from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { pipeline } from "node:stream/promises";
import { Readable } from "node:stream";

const projectId = process.env.PHOTON_PROJECT_ID;
const projectSecret = process.env.PHOTON_PROJECT_SECRET;
const port = parseInt(process.env.PHOTON_SIDECAR_PORT || "8789", 10);
const bind = process.env.PHOTON_SIDECAR_BIND || "127.0.0.1";
const sharedToken = process.env.PHOTON_SIDECAR_TOKEN;

if (!projectId || !projectSecret || !sharedToken) {
  console.error(
    "photon-sidecar: PHOTON_PROJECT_ID, PHOTON_PROJECT_SECRET and " +
      "PHOTON_SIDECAR_TOKEN must all be set."
  );
  process.exit(2);
}

// Lazy-load spectrum-ts so a missing install fails with a clear message
// instead of a cryptic module-resolution error during import.
let Spectrum, imessage, attachment, voice, effect, typing, reaction;
try {
  ({ Spectrum, attachment, voice, typing, reaction } = await import("spectrum-ts"));
  ({ imessage, effect } = await import("spectrum-ts/providers/imessage"));
} catch (e) {
  console.error(
    "photon-sidecar: spectrum-ts is not installed. Run `npm install` " +
      "inside plugins/platforms/photon/sidecar/. Original error: " +
      (e && e.stack ? e.stack : String(e))
  );
  process.exit(3);
}

const app = await Spectrum({
  projectId,
  projectSecret,
  providers: [imessage.config()],
});

// The spectrum-ts iMessage provider instance bound to this app. Exposes the
// high-level actions we use: space()/user() resolution and read receipts.
const provider = imessage(app);

// Send read receipts for inbound messages we handle (so the human sees
// Delivered → Read). On by default; set PHOTON_READ_RECEIPTS=false to disable.
const SEND_READ_RECEIPTS =
  (process.env.PHOTON_READ_RECEIPTS || "true").toLowerCase() !== "false";

async function markRead(space, message) {
  if (!SEND_READ_RECEIPTS) return;
  try {
    // The read receipt is a Space/Message *sugar* — `space.read(message)`
    // (sends `read(message)` through the space). It lives on the live Space and
    // Message objects from the inbound stream, NOT on the provider, so call it
    // there (prefer the space; fall back to the message's own `read()`).
    if (space && typeof space.read === "function") {
      await space.read(message);
    } else if (message && typeof message.read === "function") {
      await message.read();
    } else {
      console.error("photon-sidecar: no read() on space/message — receipt skipped");
    }
  } catch (e) {
    // Best-effort — SMS/some services don't support read receipts.
    console.error(
      "photon-sidecar: read receipt failed: " + (e && e.message ? e.message : e)
    );
  }
}

// ---------------------------------------------------------------------------
// Inbound bridge (sidecar -> adapter).
//
// We consume spectrum-ts' `app.messages` gRPC stream and forward each inbound
// message to the adapter's loopback webhook in the `{event:"messages", message}`
// shape it parses. We MUST do this here (not let Photon's cloud webhook deliver
// inbound) for attachments: spectrum-ts hands attachment content as an
// `asAttachment({ read(), stream() })` object whose byte-fetching closures are
// LOST the moment the message is JSON-serialized. So the bytes are only
// reachable from this Node process — we stream them to a temp file and forward
// the local path. The same loop also caches each live Space so outbound replies
// can resolve a real Space from just its id (spectrum-ts 1.x removed
// `app.space(id)`).
// ---------------------------------------------------------------------------
const spaceCache = new Map();
const _SPACE_CACHE_MAX = 1000;
function setSpace(id, space) {
  if (!id || !space) return;
  // Bound the cache so a long-running process with many conversations can't
  // grow it without limit (evict oldest, insertion-order).
  if (spaceCache.size >= _SPACE_CACHE_MAX && !spaceCache.has(id)) {
    spaceCache.delete(spaceCache.keys().next().value);
  }
  spaceCache.set(id, space);
}

const WEBHOOK_HOST = "127.0.0.1"; // sidecar -> adapter is always loopback
const WEBHOOK_PORT = parseInt(process.env.PHOTON_WEBHOOK_PORT || "8788", 10);
const WEBHOOK_PATH = process.env.PHOTON_WEBHOOK_PATH || "/photon/webhook";
const WEBHOOK_SECRET = process.env.PHOTON_WEBHOOK_SECRET || "";
const ALLOW = new Set(
  (process.env.PHOTON_ALLOWED_USERS || "")
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean)
);
const DOWNLOAD_DIR =
  process.env.PHOTON_DOWNLOAD_DIR || path.join(os.tmpdir(), "photon-attachments");
// Cap inbound attachment downloads so a large video (downloaded over a slow
// gRPC stream) can't block the message — and the agent's ack/reply — for
// minutes. Oversized attachments forward as metadata-only with a note.
const MAX_ATTACH_BYTES =
  parseInt(process.env.PHOTON_MAX_ATTACHMENT_MB || "32", 10) * 1024 * 1024;
const DOWNLOAD_TIMEOUT_MS =
  parseInt(process.env.PHOTON_DOWNLOAD_TIMEOUT_MS || "120000", 10);

const senderId = (m) =>
  typeof m?.sender?.id === "string"
    ? m.sender.id
    : m?.sender?.id?.phone || m?.sender?.phone || "";

// Mask a phone/handle for logs — keep enough to debug, never the full PII.
const maskHandle = (h) => {
  const s = String(h || "");
  return s.length <= 6 ? "***" : `${s.slice(0, 3)}***${s.slice(-2)}`;
};

// In-memory dedup (the adapter dedups too; this is belt-and-suspenders and also
// covers the spectrum reconnect/catch-up replay that re-emits recent messages).
const _seen = new Map();
const _DEDUP_MAX = 5000;
const _DEDUP_TTL_MS = 10 * 60 * 1000;
function isDuplicate(id) {
  const now = Date.now();
  if (_seen.size > _DEDUP_MAX) {
    for (const [k, t] of _seen) if (now - t > _DEDUP_TTL_MS) _seen.delete(k);
  }
  if (_seen.has(id)) return true;
  _seen.set(id, now);
  return false;
}

// Recent inbound messages by id. The reaction() builder needs the full target
// Message (it validates `isMessage` and reads `target.content.type`), not just
// an id — so /react resolves the message object from here (the ack flow always
// reacts to a message we just received and cached).
const _msgCache = new Map();
const _MSG_CACHE_MAX = 500;
function cacheMessage(message) {
  if (!message?.id) return;
  if (_msgCache.size >= _MSG_CACHE_MAX) {
    _msgCache.delete(_msgCache.keys().next().value);
  }
  _msgCache.set(message.id, message);
}

// Bounded-concurrency scheduler so a slow attachment download (large video)
// never head-of-line-blocks the message loop or other senders' messages.
const MAX_INFLIGHT = 8;
let _active = 0;
const _queue = [];
function schedule(fn) {
  _queue.push(fn);
  pump();
}
function pump() {
  while (_active < MAX_INFLIGHT && _queue.length) {
    const fn = _queue.shift();
    _active++;
    Promise.resolve()
      .then(fn)
      .catch((e) =>
        console.error(
          "photon-sidecar: inbound task error: " + (e && e.stack ? e.stack : e)
        )
      )
      .finally(() => {
        _active--;
        pump();
      });
  }
}

// Stream an inbound attachment's bytes to a temp file: write to `<final>.part`,
// verify the received byte count against the declared size, then atomically
// rename into place. Partial files are unlinked on any failure.
async function downloadAttachmentToFile(content) {
  await fsp.mkdir(DOWNLOAD_DIR, { recursive: true });
  // Sanitize BOTH the guid and the name before they reach the filesystem path —
  // `content.id` is attacker-controlled, so an unsanitized `../` could escape
  // DOWNLOAD_DIR via path.join. The allowlist regex keeps the path a basename.
  const safeGuid =
    String(content.id || "").replace(/[^\w.\-]+/g, "_") || crypto.randomUUID();
  const safeName = String(content.name || safeGuid).replace(/[^\w.\-]+/g, "_");
  const finalPath = path.join(DOWNLOAD_DIR, `${safeGuid}-${safeName}`);
  const partPath = `${finalPath}.part`;
  let received = 0;
  const ac = new AbortController();
  const timer = setTimeout(() => ac.abort(), DOWNLOAD_TIMEOUT_MS);
  try {
    // `stream()` yields a WHATWG ReadableStream of the primaryChunk bytes.
    const webStream = await content.stream();
    const nodeReadable = Readable.fromWeb(webStream);
    nodeReadable.on("data", (c) => {
      received += c.length;
      // Streaming cap: a sender can under-declare `size`, so enforce the
      // ceiling on actual bytes too — abort before exhausting disk.
      if (received > MAX_ATTACH_BYTES) ac.abort();
    });
    await pipeline(nodeReadable, fs.createWriteStream(partPath), {
      signal: ac.signal,
    });
    if (
      typeof content.size === "number" &&
      content.size > 0 &&
      received !== content.size
    ) {
      throw new Error(
        `byte-count mismatch: received ${received}, declared ${content.size}`
      );
    }
    await fsp.rename(partPath, finalPath);
    return { localPath: finalPath, bytes: received };
  } catch (e) {
    await fsp.rm(partPath, { force: true }).catch(() => {});
    throw e;
  } finally {
    clearTimeout(timer);
  }
}

async function forwardInbound(message) {
  const url = `http://${WEBHOOK_HOST}:${WEBHOOK_PORT}${WEBHOOK_PATH}`;
  const bodyStr = JSON.stringify({ event: "messages", message });
  const headers = { "Content-Type": "application/json" };
  // Sign when a webhook secret is configured so the adapter's verify_signature
  // accepts the loopback delivery: v0=HMAC_SHA256(secret, "v0:{ts}:{body}").
  if (WEBHOOK_SECRET) {
    const ts = Math.floor(Date.now() / 1000).toString();
    const sig =
      "v0=" +
      crypto
        .createHmac("sha256", WEBHOOK_SECRET)
        .update(`v0:${ts}:${bodyStr}`)
        .digest("hex");
    headers["X-Spectrum-Timestamp"] = ts;
    headers["X-Spectrum-Signature"] = sig;
  }
  try {
    // Bound the forward so a hung adapter webhook can't pin a concurrency slot.
    await fetch(url, {
      method: "POST",
      headers,
      body: bodyStr,
      signal: AbortSignal.timeout(10000),
    });
  } catch (e) {
    console.error(
      "photon-sidecar: inbound forward failed: " + (e && e.message ? e.message : e)
    );
  }
}

async function handleInbound(space, message) {
  const id = message?.id;
  if (!id || isDuplicate(id)) return;
  const from = senderId(message);
  if (ALLOW.size && from && !ALLOW.has(from)) {
    console.error(
      `photon-sidecar: drop non-allowlisted sender ${maskHandle(from)}`
    );
    return;
  }
  // Fire-and-forget read receipt (handles its own errors); don't delay forward.
  markRead(space, message);
  cacheMessage(message); // so /react can resolve this as a tapback target
  const content = message?.content || {};

  if (content.type === "attachment" && typeof content.stream === "function") {
    // Download here — the read()/stream() closures cannot survive JSON.
    let localPath = null;
    let bytes = null;
    let tooLarge = false;
    if (typeof content.size === "number" && content.size > MAX_ATTACH_BYTES) {
      // Skip the download entirely so a huge clip never blocks the agent's
      // ack/reply; forward metadata-only with a flag so the user still gets a
      // prompt response.
      tooLarge = true;
      console.error(
        `photon-sidecar: attachment ${id} too large ` +
          `(${content.size} > ${MAX_ATTACH_BYTES}) — forwarding metadata only`
      );
    } else {
      try {
        ({ localPath, bytes } = await downloadAttachmentToFile(content));
      } catch (e) {
        console.error(
          `photon-sidecar: attachment download failed for ${id}: ` +
            (e && e.message ? e.message : e)
        );
      }
    }
    return forwardInbound({
      ...message,
      content: {
        type: "attachment",
        id: content.id,
        name: content.name,
        mimeType: content.mimeType,
        size: content.size,
        localPath,
        bytes,
        tooLarge,
      },
    });
  }

  if (content.type === "reaction") {
    // Tapback / emoji reaction — forward so the adapter can surface it.
    return forwardInbound({
      ...message,
      content: {
        type: "reaction",
        emoji: content.emoji,
        target: content.target,
      },
    });
  }

  // text / other — JSON.stringify silently drops any non-serializable fields.
  return forwardInbound(message);
}

(async () => {
  try {
    for await (const [space, message] of app.messages) {
      if (space?.id) setSpace(space.id, space);
      if (message?.space?.id && space) setSpace(message.space.id, space);
      schedule(() => handleInbound(space, message));
    }
  } catch (e) {
    console.error(
      "photon-sidecar: inbound stream errored: " +
        (e && e.stack ? e.stack : String(e))
    );
  }
})();

// Control/metadata messages are tiny; cap the body so a compromised local peer
// can't OOM the sidecar by streaming an unbounded request (defence-in-depth on
// the loopback channel).
const MAX_BODY_BYTES = 2 * 1024 * 1024; // 2 MiB
async function readBody(req) {
  const chunks = [];
  let size = 0;
  for await (const chunk of req) {
    size += chunk.length;
    if (size > MAX_BODY_BYTES) {
      req.destroy();
      throw new Error("request body too large");
    }
    chunks.push(chunk);
  }
  const raw = Buffer.concat(chunks).toString("utf-8");
  if (!raw) return {};
  try {
    return JSON.parse(raw);
  } catch (e) {
    throw new Error("invalid JSON body");
  }
}

function unauthorized(res) {
  res.statusCode = 401;
  res.setHeader("Content-Type", "application/json");
  res.end(JSON.stringify({ ok: false, error: "unauthorized" }));
}

function badRequest(res, msg) {
  res.statusCode = 400;
  res.setHeader("Content-Type", "application/json");
  res.end(JSON.stringify({ ok: false, error: msg }));
}

function serverError(res) {
  res.statusCode = 500;
  res.setHeader("Content-Type", "application/json");
  // Don't leak stack traces or raw exception text to the caller — even
  // though we listen on loopback, the supervisor logs the real error
  // and the client only needs a generic failure signal.
  res.end(JSON.stringify({ ok: false, error: "internal sidecar error" }));
}

function ok(res, data) {
  res.statusCode = 200;
  res.setHeader("Content-Type", "application/json");
  res.end(JSON.stringify({ ok: true, ...data }));
}

// Outbound sends occasionally race a gRPC channel that idled out while the
// agent was thinking; Photon surfaces this as a one-off `Connection dropped` /
// gRPC UNAVAILABLE. The next call re-establishes the channel, so retry such
// transient failures with capped exponential backoff + jitter (the idiomatic
// gRPC mitigation), and only for retryable codes — never for permanent ones
// like PERMISSION_DENIED ("Target not allowed for this project").
const _RETRYABLE_GRPC = new Set([14, 4, 8]); // UNAVAILABLE, DEADLINE_EXCEEDED, RESOURCE_EXHAUSTED
const _RETRYABLE_MSG =
  /connection dropped|unavailable|deadline exceeded|resource exhausted|econnreset|socket hang up|goaway/i;
const isTransient = (e) =>
  _RETRYABLE_GRPC.has(e?.grpcCode) || _RETRYABLE_MSG.test(e?.message || "");

async function withRetry(fn, { tries = 4, baseMs = 250, capMs = 4000 } = {}) {
  let lastErr;
  for (let i = 0; i < tries; i++) {
    try {
      return await fn();
    } catch (e) {
      lastErr = e;
      if (i >= tries - 1 || !isTransient(e)) throw e;
      const backoff = Math.min(baseMs * 2 ** i, capMs);
      await new Promise((r) => setTimeout(r, backoff + Math.random() * backoff * 0.2));
    }
  }
  throw lastErr;
}

async function resolveSpace(spaceId) {
  // spectrum-ts dropped the `app.space(id)` lookup, and `space.send()` needs a
  // real resolved Space. Prefer the live Space cached from the inbound stream
  // (an outbound reply always follows an inbound that populated the cache).
  // Fall back for DMs by rebuilding the Space from the phone embedded in the id
  // (iMessage DM ids are `any;-;+E164`).
  const cached = spaceCache.get(spaceId);
  if (cached) return cached;
  const dm = spaceId.match(/;-;(\+\d+)/);
  if (dm) {
    const space = await provider.space(await provider.user({ phone: dm[1] }));
    setSpace(spaceId, space);
    return space;
  }
  throw new Error(`unable to resolve space id ${spaceId}`);
}

// Resolve a friendly screen/bubble effect name (e.g. "confetti", "slam",
// "invisible") to Apple's reverse-DNS effect id. Names + ids come straight from
// the provider's own effect map so we never hardcode the (long, churn-prone)
// identifiers. Returns null for an unknown/empty name (send proceeds without
// an effect rather than failing the message).
function resolveEffect(name) {
  if (!name) return null;
  const map = (imessage && imessage.effect && imessage.effect.message) || {};
  const id = map[String(name).trim().toLowerCase()];
  if (!id) console.error(`photon-sidecar: unknown effect "${name}" — ignoring`);
  return id || null;
}

// Wrap outgoing content with an effect when one was requested and resolved.
function withEffect(content, effectName) {
  const id = resolveEffect(effectName);
  return id && typeof effect === "function" ? effect(content, id) : content;
}

// Constant-time token comparison (don't leak the token via `!==` timing).
const _tokenBuf = Buffer.from(sharedToken);
function tokenOk(header) {
  if (typeof header !== "string") return false;
  const h = Buffer.from(header);
  // Length check first — timingSafeEqual throws on length mismatch; the token
  // length is not secret.
  return h.length === _tokenBuf.length && crypto.timingSafeEqual(h, _tokenBuf);
}

const server = http.createServer(async (req, res) => {
  if (!tokenOk(req.headers["x-hermes-sidecar-token"])) {
    return unauthorized(res);
  }
  if (req.method !== "POST") {
    res.statusCode = 405;
    return res.end();
  }
  try {
    if (req.url === "/healthz") {
      return ok(res, {});
    }
    if (req.url === "/shutdown") {
      ok(res, {});
      setTimeout(() => process.kill(process.pid, "SIGTERM"), 50);
      return;
    }
    const body = await readBody(req);
    if (req.url === "/send") {
      const { spaceId, text, effect: effectName } = body || {};
      if (!spaceId || typeof text !== "string") {
        return badRequest(res, "spaceId and text are required");
      }
      const space = await resolveSpace(spaceId);
      // spectrum-ts send() is variadic-content: a positional options object is
      // treated as a second content item and throws `c.build is not a function`.
      // Threaded replies in 1.x need the reply() builder + the original Message,
      // which the adapter doesn't hand us, so replyTo is not forwarded here.
      //
      // Plain URLs in `text` get native iMessage link previews automatically
      // (Apple's client renders the Open Graph card), so no special handling is
      // needed for rich links. An optional `effect` applies a screen/bubble
      // effect (confetti, slam, invisible ink, …).
      const result = await withRetry(() => space.send(withEffect(text, effectName)));
      return ok(res, { messageId: result?.id || result?.messageId || null });
    }
    if (req.url === "/send-attachment") {
      const { spaceId, path, name, mimeType, caption, kind, effect: effectName } =
        body || {};
      if (!spaceId || typeof path !== "string" || !path) {
        return badRequest(res, "spaceId and path are required");
      }
      const space = await resolveSpace(spaceId);

      // spectrum-ts infers name + MIME from the file extension; pass
      // overrides only when Hermes supplied them so a known-good inference
      // isn't clobbered with an empty string. Images/GIFs/videos and voice
      // (kind="voice") render natively; a sticker is just a (sticker-sized)
      // image attachment. (Documents/PDFs aren't a reliable send over this
      // line, so there's no send_document override on the adapter.)
      const opts = {};
      if (name) opts.name = name;
      if (mimeType) opts.mimeType = mimeType;
      const builder =
        kind === "voice"
          ? voice(path, Object.keys(opts).length ? opts : undefined)
          : attachment(path, Object.keys(opts).length ? opts : undefined);

      const result = await withRetry(() => space.send(withEffect(builder, effectName)));

      // iMessage delivers the caption as a separate bubble; send it
      // after the media so the attachment renders first.
      if (caption && typeof caption === "string") {
        try {
          await withRetry(() => space.send(caption));
        } catch (e) {
          console.error(
            "photon-sidecar: attachment sent but caption failed: " +
              (e && e.stack ? e.stack : String(e))
          );
        }
      }
      return ok(res, { messageId: result?.id || result?.messageId || null });
    }
    if (req.url === "/send-attachments") {
      // Multiple files in one logical send. iMessage has no atomic multi-file
      // bubble over this path, so we send them sequentially with a short pace
      // between each (back-to-back sends can arrive out of order or get
      // coalesced); `pacingMs` overrides the default.
      const { spaceId, paths, caption, pacingMs } = body || {};
      if (!spaceId || !Array.isArray(paths) || paths.length === 0) {
        return badRequest(res, "spaceId and a non-empty paths[] are required");
      }
      const space = await resolveSpace(spaceId);
      const delay = Number.isFinite(pacingMs) ? Math.max(0, pacingMs) : 600;
      const messageIds = [];
      for (let i = 0; i < paths.length; i++) {
        const p = paths[i];
        if (typeof p !== "string" || !p) continue;
        const r = await withRetry(() => space.send(attachment(p)));
        messageIds.push(r?.id || r?.messageId || null);
        if (i < paths.length - 1 && delay) {
          await new Promise((rs) => setTimeout(rs, delay));
        }
      }
      if (caption && typeof caption === "string") {
        try {
          await withRetry(() => space.send(caption));
        } catch (e) {
          console.error(
            "photon-sidecar: multi-attachment caption failed: " +
              (e && e.stack ? e.stack : String(e))
          );
        }
      }
      return ok(res, { messageIds });
    }
    if (req.url === "/typing") {
      const { spaceId, state } = body || {};
      if (!spaceId) return badRequest(res, "spaceId is required");
      const space = await resolveSpace(spaceId);
      // spectrum-ts 1.x has no `space.typing()` method — the typing indicator
      // is content: `space.send(typing("start" | "stop"))`. The gateway's
      // _keep_typing loop re-sends "start" on a cadence (iMessage indicators
      // auto-expire), and stop_typing sends "stop" when the agent is done — so
      // the bubble tracks Hermes' actual compute status. Best-effort: a missed
      // typing tick must never fail the turn.
      const sig = state === "stop" ? "stop" : "start";
      try {
        if (typeof typing === "function") await space.send(typing(sig));
      } catch (e) {
        console.error(
          "photon-sidecar: typing(" + sig + ") failed: " +
            (e && e.message ? e.message : e)
        );
      }
      return ok(res, {});
    }
    if (req.url === "/react") {
      // Bot -> user tapback. `reaction(emoji, target)` resolves to
      // `messages.setReaction(...)`; native tapbacks (❤️👍👎😂‼️❓) map to
      // Apple's tapback kinds, any other emoji is sent as an emoji reaction
      // (iOS 18+). `targetMessageId` is the guid of the message to react to.
      const { spaceId, targetMessageId, reaction: rawEmoji } = body || {};
      // Strip control characters and cap length on the reaction.
      const emoji =
        typeof rawEmoji === "string"
          ? rawEmoji.replace(/\p{Cc}/gu, "").slice(0, 16)
          : "";
      if (!spaceId || !targetMessageId || !emoji) {
        return badRequest(
          res,
          "spaceId, targetMessageId and reaction are required"
        );
      }
      const space = await resolveSpace(spaceId);
      // reaction() needs the full target Message. Prefer the cached inbound
      // message; fall back to fetching it by id from the provider.
      let target = _msgCache.get(targetMessageId);
      if (!target && typeof provider.getMessage === "function") {
        try {
          target = await provider.getMessage(space, targetMessageId);
        } catch (e) {
          console.error(
            "photon-sidecar: getMessage(" + targetMessageId + ") failed: " +
              (e && e.message ? e.message : e)
          );
        }
      }
      if (!target) {
        return badRequest(res, "target message not found");
      }
      await withRetry(() => space.send(reaction(emoji, target)));
      return ok(res, {});
    }
    res.statusCode = 404;
    res.setHeader("Content-Type", "application/json");
    return res.end(JSON.stringify({ ok: false, error: "not found" }));
  } catch (e) {
    console.error(
      "photon-sidecar: handler error: " +
        (e && e.stack ? e.stack : String(e))
    );
    // serverError() intentionally returns a generic message — see its
    // body for the rationale.
    return serverError(res);
  }
});

server.listen(port, bind, () => {
  console.error(`photon-sidecar: listening on ${bind}:${port}`);
});

// Age-based sweeper for downloaded attachments — they're consumed by the agent
// within seconds, so anything older than the TTL is safe to delete. Keeps the
// download dir from growing without bound (the OS tmp reaper is only a backstop).
const ATTACH_TTL_MS = parseInt(
  process.env.PHOTON_ATTACHMENT_TTL_MS || "3600000", // 1h
  10
);
async function sweepDownloads() {
  let names;
  try {
    names = await fsp.readdir(DOWNLOAD_DIR);
  } catch {
    return; // dir not created yet
  }
  const now = Date.now();
  for (const name of names) {
    const fp = path.join(DOWNLOAD_DIR, name);
    try {
      const st = await fsp.stat(fp);
      if (now - st.mtimeMs > ATTACH_TTL_MS) {
        await fsp.rm(fp, { force: true, recursive: true });
      }
    } catch {
      /* raced with another delete / vanished — ignore */
    }
  }
}
setInterval(() => {
  sweepDownloads().catch((e) =>
    console.error("photon-sidecar: sweep error: " + (e && e.message ? e.message : e))
  );
}, 10 * 60 * 1000).unref(); // every 10 min, don't keep the process alive

async function shutdown(signal) {
  console.error(`photon-sidecar: received ${signal}, stopping...`);
  try {
    await Promise.race([
      app.stop(),
      new Promise((resolve) => setTimeout(resolve, 3000)),
    ]);
  } catch (e) {
    console.error("photon-sidecar: app.stop() failed: " + String(e));
  }
  server.close(() => process.exit(0));
  setTimeout(() => process.exit(1), 500).unref();
}

process.on("SIGINT", () => shutdown("SIGINT"));
process.on("SIGTERM", () => shutdown("SIGTERM"));

// Don't let a stray promise rejection take the process down silently — inbound
// handlers already catch their own errors, so log and keep serving. (Python
// supervises and restarts on a real fatal exit.)
process.on("unhandledRejection", (reason) => {
  console.error(
    "photon-sidecar: unhandledRejection: " +
      (reason && reason.stack ? reason.stack : String(reason))
  );
});
