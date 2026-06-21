// Hermes Agent — Photon Spectrum sidecar
//
// Spawned by `plugins/platforms/photon/adapter.py` to bridge BOTH directions
// of messaging to Photon's Spectrum platform via the `spectrum-ts` SDK (the
// SDK is TypeScript-only, so a Node sidecar is unavoidable — there is no
// Python SDK and no public HTTP message API).
//
// Inbound  (gRPC -> Hermes): the SDK's `app.messages` async iterator is a
//   long-lived gRPC stream. We serialize each `[space, message]` to a
//   normalized JSON event and stream it to the Python adapter over a
//   loopback `GET /inbound` (NDJSON). We pause pulling from the stream while
//   no consumer is attached so a backlog isn't pulled-and-lost before the
//   gateway connects.
// Outbound (Hermes -> gRPC): `/send` drives `space.send(...)`; `/typing`
//   sends the documented `typing("start" | "stop")` content builder.
//
// Protocol (all requests require `X-Hermes-Sidecar-Token: ${TOKEN}`):
//   - GET  /inbound    -> 200 NDJSON stream; one JSON event per line, blank
//                         lines are heartbeats. One consumer at a time.
//   - POST /healthz     -> {"ok": true}
//   - POST /send        -> {"ok": true, "messageId": "..."}
//       body: {"spaceId": "...", "text": "...",
//              "format": "text" | "markdown" (default "text")}
//   - POST /send-attachment -> {"ok": true, "messageId": "..."}
//       body: {"spaceId": "...", "path": "...", "name": "..." | null,
//              "mimeType": "..." | null, "caption": "..." | null,
//              "kind": "attachment" | "voice"}
//   - POST /react       -> {"ok": true, "reactionId": "..." | null}
//       body: {"spaceId": "...", "messageId": "<target msg id>",
//              "emoji": "👀"}
//   - POST /unreact     -> {"ok": true} | 400 soft failure
//       body: {"spaceId": "...", "messageId": "<target msg id>",
//              "reactionId": "..." | null (restart-recovery fallback)}
//   - POST /typing      -> {"ok": true}
//       body: {"spaceId": "...", "state": "start" | "stop"}
//   - POST /shutdown    -> {"ok": true}; then process exits
//
// On SIGINT/SIGTERM the sidecar calls `app.stop()` (3s graceful) before
// exiting. Logs go to stderr; Python supervises restart.
//
// Requires spectrum-ts 3.x — pinned exactly in package.json because the SDK
// ships breaking majors; see README "Upgrading spectrum-ts".
//
// Env vars (required):
//   PHOTON_PROJECT_ID      (== the project's spectrumProjectId)
//   PHOTON_PROJECT_SECRET
//   PHOTON_SIDECAR_PORT
//   PHOTON_SIDECAR_TOKEN
// Optional:
//   PHOTON_SIDECAR_BIND    (default 127.0.0.1)
//   PHOTON_SIDECAR_WATCH_STDIN  "1" = exit when stdin hits EOF (set by the
//                          adapter, which holds our stdin pipe — parent-death
//                          detection so a dead gateway can't orphan us)
//   PHOTON_TELEMETRY       enable Spectrum SDK telemetry ("true"/"1"/"on"/"yes";
//                          default off — toggle with `hermes photon telemetry`)

import http from "node:http";
import crypto from "node:crypto";
import { once } from "node:events";
import { patchSpectrumTs } from "./patch-spectrum-mixed-attachments.mjs";

const projectId = process.env.PHOTON_PROJECT_ID;
const projectSecret = process.env.PHOTON_PROJECT_SECRET;
const port = parseInt(process.env.PHOTON_SIDECAR_PORT || "8789", 10);
const bind = process.env.PHOTON_SIDECAR_BIND || "127.0.0.1";
const sharedToken = process.env.PHOTON_SIDECAR_TOKEN;
const telemetry = /^(1|true|yes|on)$/i.test(
  (process.env.PHOTON_TELEMETRY || "").trim()
);

// Inbound binary content is read into memory and base64-inlined on the NDJSON
// event so the Python adapter can cache the real bytes (and the agent can see
// images / transcribe voice). Cap the size we inline — above it we forward
// metadata only and the adapter surfaces a text marker, so one large clip can't
// balloon a single NDJSON line. Override via PHOTON_MAX_INLINE_ATTACHMENT_BYTES.
const MAX_INLINE_ATTACHMENT_BYTES =
  Number(process.env.PHOTON_MAX_INLINE_ATTACHMENT_BYTES) || 20 * 1024 * 1024;
const DM_CHAT_GUID_RE = /^any;-;(\+\d{6,})$/;
const E164_RE = /^\+\d{6,}$/;
const MAX_KNOWN_SPACES = 2048;
const MAX_KNOWN_MESSAGES = 1024;
const MAX_REACTION_HANDLES = 512;

if (!projectId || !projectSecret || !sharedToken) {
  console.error(
    "photon-sidecar: PHOTON_PROJECT_ID, PHOTON_PROJECT_SECRET and " +
      "PHOTON_SIDECAR_TOKEN must all be set."
  );
  process.exit(2);
}

// Lazy-load spectrum-ts so a missing install fails with a clear message
// instead of a cryptic module-resolution error during import. Apply Hermes'
// pinned-sdk compatibility patch first so existing installs self-heal at
// runtime, not only during npm postinstall.
try {
  const patchResult = patchSpectrumTs();
  if (patchResult.patched) {
    console.error(
      `photon-sidecar: spectrum mixed attachment patch applied: ${patchResult.file}`
    );
  }
} catch (e) {
  console.error(
    "photon-sidecar: spectrum mixed attachment patch failed. " +
      "Run `npm install` inside plugins/platforms/photon/sidecar/ or " +
      "upgrade the Photon sidecar patch for the pinned spectrum-ts version. " +
      "Original error: " +
      (e && e.stack ? e.stack : String(e))
  );
  process.exit(3);
}
let Spectrum,
  imessage,
  attachment,
  voice,
  spectrumText,
  spectrumMarkdown,
  spectrumTyping;
try {
  ({
    Spectrum,
    attachment,
    voice,
    text: spectrumText,
    markdown: spectrumMarkdown,
    typing: spectrumTyping,
  } = await import("spectrum-ts"));
  ({ imessage } = await import("spectrum-ts/providers/imessage"));
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
  options: { flattenGroups: true },
  telemetry,
});

// ---------------------------------------------------------------------------
// Inbound: forward `app.messages` (gRPC stream) to the Python consumer.

// At most one Python consumer is attached at a time (the gateway adapter).
let consumerRes = null;
let consumerWaiters = [];
const knownSpaces = new Map();
// Inbound Message objects by id, so /react can usually skip a
// `space.getMessage` round trip when tapping back on a recent message.
const knownMessages = new Map();
// One reaction handle per reacted-to message (key `${spaceId}\0${messageId}`,
// value {emoji, handle}) — mirrors iMessage's one-tapback-per-sender
// semantics; a new /react on the same target overwrites the slot. The handle
// is the outbound reaction Message returned by `target.react()`, kept so
// /unreact can `unsend()` it later.
const reactionHandles = new Map();

function lruSet(map, key, value, cap) {
  if (map.has(key)) map.delete(key);
  map.set(key, value);
  if (map.size > cap) {
    const oldest = map.keys().next().value;
    if (oldest !== undefined) map.delete(oldest);
  }
}

function rememberKnownSpace(id, space) {
  if (!id || typeof id !== "string" || !space) return;
  lruSet(knownSpaces, id, space, MAX_KNOWN_SPACES);
}

function rememberKnownMessage(message) {
  const id = message?.id;
  if (!id || typeof id !== "string") return;
  lruSet(knownMessages, id, message, MAX_KNOWN_MESSAGES);
}

function phoneTargetFromSpaceId(spaceId) {
  if (typeof spaceId !== "string") return null;
  if (E164_RE.test(spaceId)) return spaceId;
  const dmGuid = spaceId.match(DM_CHAT_GUID_RE);
  return dmGuid ? dmGuid[1] : null;
}

function rememberInboundSpace(space, message) {
  const msgSpace = message?.space || {};
  const ids = [space?.id, msgSpace.id];
  for (const id of ids) {
    rememberKnownSpace(id, space);
    const phone = phoneTargetFromSpaceId(id);
    if (phone) rememberKnownSpace(phone, space);
  }
}

function waitForConsumer() {
  if (consumerRes) return Promise.resolve();
  return new Promise((resolve) => consumerWaiters.push(resolve));
}

function setConsumer(res) {
  consumerRes = res;
  const waiters = consumerWaiters;
  consumerWaiters = [];
  for (const resolve of waiters) resolve();
}

function clearConsumer(res) {
  if (consumerRes === res) consumerRes = null;
}

// Write one NDJSON line to the active consumer. Blocks until a consumer is
// connected; if the write fails (consumer vanished mid-flight) we wait for a
// new consumer and retry, so a message is never silently dropped here.
async function deliver(line) {
  for (;;) {
    await waitForConsumer();
    const res = consumerRes;
    if (!res) continue;
    try {
      const flushed = res.write(line + "\n");
      if (!flushed) await once(res, "drain");
      return;
    } catch {
      clearConsumer(res);
    }
  }
}

async function normalizeBinaryContent(content) {
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

  // Read the bytes eagerly and base64-inline them as `data` so the Python
  // adapter can cache the real file (the agent then sees images and can run
  // STT on voice notes). Spectrum content objects may not outlive this stream
  // iteration, so a lazy/on-demand fetch isn't safe. Over-cap content (when
  // size is known up front) is forwarded as metadata only and the adapter falls
  // back to a text marker. A read failure must never break the inbound loop.
  const label = `${content.type} ${meta.name ?? meta.id ?? "(unnamed)"}`;
  if (meta.size !== null && meta.size > MAX_INLINE_ATTACHMENT_BYTES) {
    console.error(
      `photon-sidecar: ${label} (${meta.size} bytes) ` +
        `exceeds inline cap ${MAX_INLINE_ATTACHMENT_BYTES}; forwarding metadata only`
    );
    return meta;
  }
  if (typeof content.read === "function") {
    try {
      const buf = await content.read();
      // Guard the case where size was unknown but the bytes turn out to be
      // over the cap.
      if (buf && buf.length > MAX_INLINE_ATTACHMENT_BYTES) {
        console.error(
          `photon-sidecar: ${label} (${buf.length} bytes) ` +
            `exceeds inline cap after read; forwarding metadata only`
        );
        return meta;
      }
      meta.data = Buffer.from(buf).toString("base64");
      meta.encoding = "base64";
    } catch (e) {
      console.error(
        `photon-sidecar: failed to read ${content.type} bytes ` +
          "(forwarding metadata only): " +
          (e && e.stack ? e.stack : String(e))
      );
    }
  }
  return meta;
}

async function normalizeContent(content) {
  if (!content || typeof content !== "object") {
    return { type: "unknown" };
  }
  if (content.type === "text") {
    return { type: "text", text: content.text || "" };
  }
  if (content.type === "attachment" || content.type === "voice") {
    return await normalizeBinaryContent(content);
  }
  if (content.type === "group") {
    const items = [];
    for (const item of Array.isArray(content.items) ? content.items : []) {
      items.push({
        id: item && typeof item === "object" ? item.id ?? null : null,
        content: await normalizeContent(item?.content),
      });
    }
    return { type: "group", items };
  }
  if (content.type === "reaction") {
    return {
      type: "reaction",
      emoji: content.emoji || "",
      targetMessageId: content.target?.id ?? null,
      // Lets Python gate "is this a reaction to one of MY messages" without
      // tracking every outbound id. May be null if the provider doesn't
      // hydrate the target — Python falls back to its own sent-id cache.
      targetDirection: content.target?.direction ?? null,
    };
  }
  return { type: content.type || "unknown" };
}

async function normalizeEvent(space, message) {
  try {
    const msgSpace = message.space || {};
    const ts = message.timestamp;
    return {
      messageId: message.id ?? null,
      platform: message.platform || space.__platform || "iMessage",
      space: {
        id: space.id ?? msgSpace.id ?? null,
        // iMessage spaces carry `type` ("dm"|"group") and `phone` directly.
        type: space.type ?? msgSpace.type ?? "dm",
        phone: space.phone ?? msgSpace.phone ?? null,
      },
      sender: { id: message.sender ? message.sender.id : null },
      content: await normalizeContent(message.content),
      timestamp:
        ts instanceof Date ? ts.toISOString() : ts ? String(ts) : null,
    };
  } catch (e) {
    console.error(
      "photon-sidecar: failed to normalize inbound message: " + String(e)
    );
    return null;
  }
}

// spectrum-ts handles in-session gRPC reconnects internally, but if the async
// iterator itself throws or ends, this consumer would stop forever. Wrap it in
// a re-subscribe loop with capped exponential backoff + jitter so inbound
// always recovers (the adapter dedupes any catch-up replay).
(async () => {
  let backoff = 1000;
  for (;;) {
    try {
      for await (const [space, message] of app.messages) {
        backoff = 1000; // healthy traffic — reset
        // Only forward inbound messages (ignore our own outbound echoes).
        if (message && message.direction && message.direction !== "inbound") {
          continue;
        }
        rememberInboundSpace(space, message);
        rememberKnownMessage(message);
        const event = await normalizeEvent(space, message);
        if (!event) continue;
        await deliver(JSON.stringify(event));
      }
      console.error("photon-sidecar: inbound stream ended — re-subscribing");
    } catch (e) {
      console.error(
        "photon-sidecar: inbound stream errored — restarting: " +
          (e && e.message ? e.message : String(e))
      );
    }
    await new Promise((r) =>
      setTimeout(r, backoff + Math.random() * backoff * 0.2)
    );
    backoff = Math.min(backoff * 2, 30000);
  }
})();

// ---------------------------------------------------------------------------
// HTTP control + inbound server (loopback only).

// Control-message bodies are tiny; cap the body so a compromised local peer
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

function handleInbound(req, res) {
  res.statusCode = 200;
  res.setHeader("Content-Type", "application/x-ndjson");
  res.setHeader("Cache-Control", "no-store");
  res.setHeader("Connection", "keep-alive");
  // One consumer at a time — a fresh connection (e.g. after a reconnect)
  // supersedes the previous one.
  if (consumerRes && consumerRes !== res) {
    try {
      consumerRes.end();
    } catch {
      /* ignore */
    }
  }
  setConsumer(res);
  // Heartbeat keeps the socket warm through idle periods and lets the Python
  // side detect a dead pipe promptly.
  const heartbeat = setInterval(() => {
    try {
      res.write("\n");
    } catch {
      /* ignore */
    }
  }, 25000);
  const cleanup = () => {
    clearInterval(heartbeat);
    clearConsumer(res);
  };
  req.on("close", cleanup);
  req.on("aborted", cleanup);
  res.on("error", cleanup);
}

async function resolveSpace(spaceId) {
  const cached = knownSpaces.get(spaceId);
  if (cached) return cached;

  const im = imessage(app);
  const phoneTarget = phoneTargetFromSpaceId(spaceId);
  let space = null;

  // A bare E.164 phone number addresses a DM, so callers can pass just
  // "+1..." (e.g. PHOTON_HOME_CHANNEL for cron delivery) instead of an opaque
  // inbound space id. Photon also represents DM chat ids as `any;-;+1...`;
  // normalize those through the same path and resolve the phone to a user
  // before creating the DM space.
  if (phoneTarget) {
    try {
      const user = await im.user(phoneTarget);
      space = await im.space.create(user);
    } catch (e) {
      console.error(
        "photon-sidecar: phone->DM space.create failed: " +
          (e && e.stack ? e.stack : String(e))
      );
    }
  }
  // Anything else — typically an opaque group GUID — is rehydrated from the
  // persisted id via `space.get`, so group spaces stay reachable after a
  // sidecar restart even before any fresh inbound message in that group.
  if (!space) {
    try {
      space = await im.space.get(spaceId);
    } catch (e) {
      console.error(
        "photon-sidecar: space.get failed: " +
          (e && e.stack ? e.stack : String(e))
      );
    }
  }
  if (!space) throw new Error(`unable to resolve space id ${spaceId}`);

  rememberKnownSpace(spaceId, space);
  if (phoneTarget) rememberKnownSpace(phoneTarget, space);
  rememberKnownSpace(space?.id, space);
  return space;
}

// Constant-time token comparison — don't leak the token via `!==` timing.
const _tokenBuf = Buffer.from(sharedToken);
function tokenOk(header) {
  if (typeof header !== "string") return false;
  const h = Buffer.from(header);
  return h.length === _tokenBuf.length && crypto.timingSafeEqual(h, _tokenBuf);
}

const server = http.createServer(async (req, res) => {
  if (!tokenOk(req.headers["x-hermes-sidecar-token"])) {
    return unauthorized(res);
  }
  // Long-lived inbound NDJSON stream.
  if (req.method === "GET" && req.url === "/inbound") {
    return handleInbound(req, res);
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
      const { spaceId, text, format = "text" } = body || {};
      if (!spaceId || typeof text !== "string") {
        return badRequest(res, "spaceId and text are required");
      }
      if (format !== "text" && format !== "markdown") {
        return badRequest(res, "format must be text or markdown");
      }
      const space = await resolveSpace(spaceId);
      // iMessage renders markdown natively; spectrum-ts degrades it to
      // readable plain text on platforms that don't.
      const builder =
        format === "markdown" ? spectrumMarkdown(text) : spectrumText(text);
      const result = await space.send(builder);
      return ok(res, { messageId: result?.id || null });
    }
    if (req.url === "/send-attachment") {
      const { spaceId, path, name, mimeType, caption, kind } =
        body || {};
      if (!spaceId || typeof path !== "string" || !path) {
        return badRequest(res, "spaceId and path are required");
      }
      const space = await resolveSpace(spaceId);

      // spectrum-ts infers name + MIME from the file extension; pass
      // overrides only when Hermes supplied them so a known-good
      // inference isn't clobbered with an empty string.
      const opts = {};
      if (name) opts.name = name;
      if (mimeType) opts.mimeType = mimeType;
      const builder =
        kind === "voice"
          ? voice(path, Object.keys(opts).length ? opts : undefined)
          : attachment(path, Object.keys(opts).length ? opts : undefined);

      const result = await space.send(builder);

      // iMessage delivers the caption as a separate bubble; send it
      // after the media so the attachment renders first.
      if (caption && typeof caption === "string") {
        try {
          await space.send(spectrumText(caption));
        } catch (e) {
          console.error(
            "photon-sidecar: attachment sent but caption failed: " +
              (e && e.stack ? e.stack : String(e))
          );
        }
      }
      return ok(res, { messageId: result?.id || null });
    }
    if (req.url === "/react") {
      const { spaceId, messageId, emoji } = body || {};
      if (!spaceId || !messageId || typeof emoji !== "string" || !emoji) {
        return badRequest(res, "spaceId, messageId and emoji are required");
      }
      const space = await resolveSpace(spaceId);
      const target =
        knownMessages.get(messageId) ?? (await space.getMessage(messageId));
      if (!target) {
        return badRequest(res, "message not found");
      }
      const handle = await target.react(emoji);
      if (!handle) {
        return badRequest(res, "reactions not supported on this platform");
      }
      lruSet(
        reactionHandles,
        `${spaceId}\u0000${messageId}`,
        { emoji, handle },
        MAX_REACTION_HANDLES
      );
      return ok(res, { reactionId: handle.id ?? null });
    }
    if (req.url === "/unreact") {
      const { spaceId, messageId, reactionId } = body || {};
      if (!spaceId || !messageId) {
        return badRequest(res, "spaceId and messageId are required");
      }
      const key = `${spaceId}\u0000${messageId}`;
      const slot = reactionHandles.get(key);
      if (slot) {
        await slot.handle.unsend();
        reactionHandles.delete(key);
        return ok(res, {});
      }
      // Restart-recovery: the live handle is gone, so try rehydrating the
      // reaction message by id and retracting it. Only outbound messages can
      // be unsent — if the provider rehydrates it as inbound (or not at all)
      // this throws, and that's an expected soft failure, not a sidecar bug:
      // a stale tapback self-heals when the next /react replaces it.
      if (reactionId) {
        try {
          const space = await resolveSpace(spaceId);
          const msg = await space.getMessage(reactionId);
          if (msg) {
            await space.unsend(msg);
            return ok(res, {});
          }
        } catch (e) {
          console.error(
            "photon-sidecar: best-effort unreact failed: " +
              (e && e.message ? e.message : String(e))
          );
        }
        return badRequest(res, "reaction not removable");
      }
      return badRequest(res, "no tracked reaction for message");
    }
    if (req.url === "/typing") {
      const { spaceId, state = "start" } = body || {};
      if (!spaceId) return badRequest(res, "spaceId is required");
      if (state !== "start" && state !== "stop") {
        return badRequest(res, "state must be start or stop");
      }
      const space = await resolveSpace(spaceId);
      await space.send(spectrumTyping(state));
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

let stopping = false;
async function shutdown(signal) {
  // Re-entry guard: stdin EOF, a signal and /shutdown can all fire together
  // during one teardown.
  if (stopping) return;
  stopping = true;
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

// Lifetime binding to the parent. The adapter spawns us with stdin as a pipe
// it holds open; EOF means the gateway process is gone — including hard
// deaths (crash, SIGKILL) where no signal and no /shutdown ever reaches us.
// Without this, an orphaned sidecar squats the port and keeps consuming the
// inbound gRPC stream, and every replacement spawn dies on EADDRINUSE.
// Opt-in via env so manual `node index.mjs` runs aren't affected.
if (process.env.PHOTON_SIDECAR_WATCH_STDIN === "1") {
  process.stdin.resume();
  process.stdin.on("end", () => shutdown("stdin EOF (parent exited)"));
  process.stdin.on("error", () => shutdown("stdin error (parent exited)"));
}

// Don't let a stray promise rejection take the process down silently — handlers
// catch their own errors, so log and keep serving (Python supervises restart on
// a real fatal exit).
process.on("unhandledRejection", (reason) => {
  console.error(
    "photon-sidecar: unhandledRejection: " +
      (reason && reason.stack ? reason.stack : String(reason))
  );
});
