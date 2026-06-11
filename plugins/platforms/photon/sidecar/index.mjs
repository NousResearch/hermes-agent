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
//       body: {"spaceId": "...", "text": "..."}
//   - POST /send-attachment -> {"ok": true, "messageId": "..."}
//       body: {"spaceId": "...", "path": "...", "name": "..." | null,
//              "mimeType": "..." | null, "caption": "..." | null,
//              "kind": "attachment" | "voice"}
//   - POST /typing      -> {"ok": true}
//       body: {"spaceId": "...", "state": "start" | "stop"}
//   - POST /shutdown    -> {"ok": true}; then process exits
//
// On SIGINT/SIGTERM the sidecar calls `app.stop()` (3s graceful) before
// exiting. Logs go to stderr; Python supervises restart.
//
// Env vars (required):
//   PHOTON_PROJECT_ID      (== the project's spectrumProjectId)
//   PHOTON_PROJECT_SECRET
//   PHOTON_SIDECAR_PORT
//   PHOTON_SIDECAR_TOKEN
// Optional:
//   PHOTON_SIDECAR_BIND    (default 127.0.0.1)
//   PHOTON_TELEMETRY       enable Spectrum SDK telemetry ("true"/"1"/"on"/"yes";
//                          default off — toggle with `hermes photon telemetry`)

import http from "node:http";
import crypto from "node:crypto";
import { once } from "node:events";

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

if (!projectId || !projectSecret || !sharedToken) {
  console.error(
    "photon-sidecar: PHOTON_PROJECT_ID, PHOTON_PROJECT_SECRET and " +
      "PHOTON_SIDECAR_TOKEN must all be set."
  );
  process.exit(2);
}

// Lazy-load spectrum-ts so a missing install fails with a clear message
// instead of a cryptic module-resolution error during import.
let Spectrum, imessage, attachment, voice, spectrumText, spectrumTyping;
try {
  ({
    Spectrum,
    attachment,
    voice,
    text: spectrumText,
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

function rememberKnownSpace(id, space) {
  if (!id || typeof id !== "string" || !space) return;
  if (knownSpaces.has(id)) knownSpaces.delete(id);
  knownSpaces.set(id, space);
  if (knownSpaces.size > MAX_KNOWN_SPACES) {
    const oldest = knownSpaces.keys().next().value;
    if (oldest) knownSpaces.delete(oldest);
  }
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

  const phoneTarget = phoneTargetFromSpaceId(spaceId);
  // A bare E.164 phone number addresses a DM. Resolve the user, then the (DM)
  // space — `imessage(app).user(phone)` -> `im.space(user)` — so callers can
  // pass just "+1..." (e.g. PHOTON_HOME_CHANNEL for cron delivery) instead of
  // an opaque inbound space id. Photon also represents DM chat ids as
  // `any;-;+1...`; normalize those through the same path so replies to inbound
  // DMs still resolve after Python stores the inbound `space.id`.
  if (phoneTarget && imessage) {
    try {
      const im = imessage(app);
      const user = await im.user(phoneTarget);
      const space = await im.space(user);
      rememberKnownSpace(spaceId, space);
      rememberKnownSpace(phoneTarget, space);
      rememberKnownSpace(space?.id, space);
      return space;
    } catch (e) {
      console.error(
        "photon-sidecar: phone->DM resolution failed: " +
          (e && e.stack ? e.stack : String(e))
      );
    }
  }
  // No cache hit and not a phone/DM target. spectrum-ts exposes no API to
  // rehydrate an arbitrary opaque space id: a Space is only obtained from the
  // inbound `[space, message]` stream (cached above in `knownSpaces`) or
  // reconstructed for a DM from its phone number. So a group space whose cache
  // entry was lost — e.g. after a sidecar restart with no fresh inbound message
  // in that group — cannot be resolved here; a new inbound message in the group
  // re-warms the cache. DMs are unaffected (reconstructed from the phone).
  throw new Error(`unable to resolve space id ${spaceId}`);
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
      const { spaceId, text } = body || {};
      if (!spaceId || typeof text !== "string") {
        return badRequest(res, "spaceId and text are required");
      }
      const space = await resolveSpace(spaceId);
      const result = await space.send(spectrumText(text));
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

// Don't let a stray promise rejection take the process down silently — handlers
// catch their own errors, so log and keep serving (Python supervises restart on
// a real fatal exit).
process.on("unhandledRejection", (reason) => {
  console.error(
    "photon-sidecar: unhandledRejection: " +
      (reason && reason.stack ? reason.stack : String(reason))
  );
});
