// Hermes Agent — Photon Spectrum sidecar
//
// Spawned by `plugins/platforms/photon/adapter.py` to bridge outbound
// messaging to Photon's Spectrum platform. Inbound messages go directly
// from Photon's webhook to Hermes' Python aiohttp receiver — this
// sidecar handles ONLY outbound calls (which require the spectrum-ts
// SDK because Photon has no public HTTP send endpoint today).
//
// Protocol:
//   - The sidecar listens on http://127.0.0.1:${PORT} (loopback only)
//   - Each request must include `X-Hermes-Sidecar-Token: ${TOKEN}`
//   - POST /healthz                     -> {"ok": true}
//   - POST /send                        -> {"ok": true, "messageId": "..."}
//       body: {"spaceId": "...", "text": "...", "replyTo": "..." | null}
//   - POST /send-attachment             -> {"ok": true, "messageId": "..."}
//       body: {"spaceId": "...", "path": "...", "name": "..." | null,
//              "mimeType": "..." | null, "caption": "..." | null,
//              "kind": "attachment" | "voice", "replyTo": "..." | null}
//   - POST /typing                      -> {"ok": true}
//       body: {"spaceId": "..."}
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
let Spectrum, imessage, attachment, voice;
try {
  ({ Spectrum, attachment, voice } = await import("spectrum-ts"));
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
});

// Drain the inbound stream — Photon's webhook is the canonical inbound path,
// but we still consume `app.messages` so spectrum-ts' reconnect/heartbeat logic
// keeps running.  We also cache each live Space here: spectrum-ts removed the
// `app.space(id)` lookup, so outbound replies resolve their Space from this
// cache (an outbound reply always follows an inbound that populated it).
const spaceCache = new Map();
(async () => {
  try {
    for await (const [space, message] of app.messages) {
      if (space?.id) spaceCache.set(space.id, space);
      if (message?.space?.id && space) spaceCache.set(message.space.id, space);
    }
  } catch (e) {
    console.error(
      "photon-sidecar: inbound stream errored: " +
        (e && e.stack ? e.stack : String(e))
    );
  }
})();

async function readBody(req) {
  const chunks = [];
  for await (const chunk of req) chunks.push(chunk);
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
    const im = imessage(app);
    const space = await im.space(await im.user({ phone: dm[1] }));
    spaceCache.set(spaceId, space);
    return space;
  }
  throw new Error(`unable to resolve space id ${spaceId}`);
}

const server = http.createServer(async (req, res) => {
  if (req.headers["x-hermes-sidecar-token"] !== sharedToken) {
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
      const { spaceId, text } = body || {};
      if (!spaceId || typeof text !== "string") {
        return badRequest(res, "spaceId and text are required");
      }
      const space = await resolveSpace(spaceId);
      // spectrum-ts send() is variadic-content: a positional options object is
      // treated as a second content item and throws `c.build is not a function`.
      // Threaded replies in 1.x need the reply() builder + the original Message,
      // which the adapter doesn't hand us, so replyTo is not forwarded here.
      const result = await withRetry(() => space.send(text));
      return ok(res, { messageId: result?.id || result?.messageId || null });
    }
    if (req.url === "/send-attachment") {
      const { spaceId, path, name, mimeType, caption, kind } = body || {};
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

      const result = await withRetry(() => space.send(builder));

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
    if (req.url === "/typing") {
      const { spaceId } = body || {};
      if (!spaceId) return badRequest(res, "spaceId is required");
      const space = await resolveSpace(spaceId);
      if (typeof space.typing === "function") {
        await space.typing();
      } else if (typeof space.setTyping === "function") {
        await space.setTyping(true);
      }
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
