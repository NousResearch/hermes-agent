// Hermes Agent — Wechaty sidecar
//
// Spawned by plugins/platforms/wechaty/adapter.py. Runs a Wechaty bot and
// bridges messaging to the Python gateway over loopback HTTP.
//
// Inbound:  GET /inbound  -> NDJSON stream (messages, scan QR, login state)
// Outbound: POST /send    -> { chatId, text }
//           POST /send-file -> { chatId, path, name?, caption? }
//           POST /typing  -> { chatId, state: "start"|"stop" }
//           POST /healthz -> { ok: true, loggedIn: bool }
//           POST /shutdown
//
// Auth: X-Hermes-Sidecar-Token header on every request.
//
// chatId format:
//   contact:<wechaty-contact-id>  — DM
//   room:<wechaty-room-id>          — group

import http from "node:http";
import crypto from "node:crypto";
import { once } from "node:events";
import fs from "node:fs/promises";
import path from "node:path";

const port = parseInt(process.env.WECHATY_SIDECAR_PORT || "8790", 10);
const bind = process.env.WECHATY_SIDECAR_BIND || "127.0.0.1";
const sharedToken = process.env.WECHATY_SIDECAR_TOKEN;
const botName = process.env.WECHATY_BOT_NAME || "hermes-wechaty";
const puppet = (process.env.WECHATY_PUPPET || "").trim();
const puppetToken =
  process.env.WECHATY_PUPPET_SERVICE_TOKEN ||
  process.env.WECHATY_TOKEN ||
  "";

const MAX_BODY_BYTES = 2 * 1024 * 1024;
const MAX_INLINE_ATTACHMENT_BYTES =
  Number(process.env.WECHATY_MAX_INLINE_ATTACHMENT_BYTES) || 8 * 1024 * 1024;

if (!sharedToken) {
  console.error("wechaty-sidecar: WECHATY_SIDECAR_TOKEN must be set.");
  process.exit(2);
}

let WechatyBuilder;
let ScanStatus;
let MessageType;
let FileBox;
let qrTerm;
try {
  const wechaty = await import("wechaty");
  WechatyBuilder = wechaty.WechatyBuilder;
  ScanStatus = wechaty.ScanStatus;
  MessageType = wechaty.types.Message;
  ({ FileBox } = await import("file-box"));
  qrTerm = (await import("qrcode-terminal")).default;
} catch (e) {
  console.error(
    "wechaty-sidecar: missing deps. Run `npm install` in sidecar/. Error: " +
      (e && e.stack ? e.stack : String(e))
  );
  process.exit(3);
}

// ---------------------------------------------------------------------------
// NDJSON consumer (one client at a time)

let consumerRes = null;
let consumerWaiters = [];

function waitForConsumer() {
  if (consumerRes) return Promise.resolve();
  return new Promise((resolve) => consumerWaiters.push(resolve));
}

function setConsumer(res) {
  consumerRes = res;
  for (const resolve of consumerWaiters) resolve();
  consumerWaiters = [];
}

function clearConsumer(res) {
  if (consumerRes === res) consumerRes = null;
}

async function deliver(event) {
  const line = JSON.stringify(event);
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

// ---------------------------------------------------------------------------
// Wechaty bot

const buildOptions = { name: botName };
if (puppet) {
  buildOptions.puppet = puppet;
  if (puppetToken) {
    buildOptions.puppetOptions = { token: puppetToken };
  }
} else if (puppetToken) {
  buildOptions.puppet = "wechaty-puppet-service";
  buildOptions.puppetOptions = { token: puppetToken };
}

const bot = WechatyBuilder.build(buildOptions);
let loggedIn = false;

function chatIdForMessage(message) {
  const room = message.room();
  if (room) {
    return `room:${room.id}`;
  }
  const talker = message.talker();
  return `contact:${talker.id}`;
}

async function normalizeInboundMessage(message) {
  const room = message.room();
  const talker = message.talker();
  const mentionSelf = room ? await message.mentionSelf() : false;
  const text = message.text() || "";
  const type = message.type();
  const payload = {
    type: "message",
    messageId: message.id,
    chatId: chatIdForMessage(message),
    chatType: room ? "group" : "dm",
    chatName: room ? (await room.topic()) || room.id : (await talker.name()) || talker.id,
    senderId: talker.id,
    senderName: (await talker.name()) || talker.id,
    text,
    mentionSelf,
    messageType: type,
    timestamp: new Date().toISOString(),
  };

  if (type === MessageType.Image) {
    try {
      const fileBox = await message.toFileBox();
      const buf = await fileBox.toBuffer();
      if (buf.length <= MAX_INLINE_ATTACHMENT_BYTES) {
        payload.attachment = {
          kind: "image",
          name: fileBox.name || "image.jpg",
          mimeType: fileBox.mimeType || "image/jpeg",
          data: buf.toString("base64"),
          encoding: "base64",
        };
      } else {
        payload.attachment = {
          kind: "image",
          name: fileBox.name || "image.jpg",
          mimeType: fileBox.mimeType || "image/jpeg",
          tooLarge: true,
          size: buf.length,
        };
      }
    } catch (e) {
      console.error(
        "wechaty-sidecar: failed to read image attachment: " + String(e)
      );
    }
  }

  return payload;
}

bot.on("scan", async (qrcode, status) => {
  const statusName = ScanStatus[status] || String(status);
  const qrUrl = `https://wechaty.js.org/qrcode/${encodeURIComponent(qrcode)}`;
  console.error(`wechaty-sidecar: scan status=${statusName}`);
  // Always print the URL — terminal ASCII QR is hard to scan from logs/screenshots.
  console.error(`wechaty-sidecar: open this URL for a scannable QR image:\n  ${qrUrl}`);
  try {
    qrTerm.generate(qrcode, { small: true });
  } catch {
    console.error("wechaty-sidecar: qrcode-terminal failed; use the URL above");
  }
  await deliver({
    type: "scan",
    status: statusName,
    qrcode,
    timestamp: new Date().toISOString(),
  });
});

bot.on("login", async (user) => {
  loggedIn = true;
  console.error(`wechaty-sidecar: logged in as ${user.name()} (${user.id})`);
  await deliver({
    type: "login",
    userId: user.id,
    userName: user.name(),
    timestamp: new Date().toISOString(),
  });
});

bot.on("logout", async (user) => {
  loggedIn = false;
  const name = user ? user.name() : "unknown";
  console.error(`wechaty-sidecar: logged out (${name})`);
  await deliver({
    type: "logout",
    userName: name,
    timestamp: new Date().toISOString(),
  });
});

bot.on("message", async (message) => {
  try {
    if (message.self()) return;
    const event = await normalizeInboundMessage(message);
    await deliver(event);
  } catch (e) {
    console.error(
      "wechaty-sidecar: inbound message handler failed: " +
        (e && e.stack ? e.stack : String(e))
    );
  }
});

bot.start().catch((e) => {
  console.error(
    "wechaty-sidecar: bot.start() failed: " +
      (e && e.stack ? e.stack : String(e))
  );
  process.exit(4);
});

// ---------------------------------------------------------------------------
// Target resolution

async function resolveSayTarget(chatId) {
  const idx = chatId.indexOf(":");
  if (idx <= 0) throw new Error(`invalid chatId ${chatId}`);
  const kind = chatId.slice(0, idx);
  const id = chatId.slice(idx + 1);
  if (kind === "room") {
    const room = await bot.Room.find({ id });
    if (!room) throw new Error(`room not found: ${id}`);
    return room;
  }
  if (kind === "contact") {
    const contact = await bot.Contact.find({ id });
    if (!contact) throw new Error(`contact not found: ${id}`);
    return contact;
  }
  throw new Error(`unknown chatId kind: ${kind}`);
}

// ---------------------------------------------------------------------------
// HTTP server

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
  return JSON.parse(raw);
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
  res.end(JSON.stringify({ ok: false, error: "internal sidecar error" }));
}

function ok(res, data) {
  res.statusCode = 200;
  res.setHeader("Content-Type", "application/json");
  res.end(JSON.stringify({ ok: true, ...data }));
}

const _tokenBuf = Buffer.from(sharedToken);
function tokenOk(header) {
  if (typeof header !== "string") return false;
  const h = Buffer.from(header);
  return h.length === _tokenBuf.length && crypto.timingSafeEqual(h, _tokenBuf);
}

function handleInbound(req, res) {
  res.statusCode = 200;
  res.setHeader("Content-Type", "application/x-ndjson");
  res.setHeader("Cache-Control", "no-store");
  res.setHeader("Connection", "keep-alive");
  if (consumerRes && consumerRes !== res) {
    try {
      consumerRes.end();
    } catch {
      /* ignore */
    }
  }
  setConsumer(res);
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

const server = http.createServer(async (req, res) => {
  if (!tokenOk(req.headers["x-hermes-sidecar-token"])) {
    return unauthorized(res);
  }
  if (req.method === "GET" && req.url === "/inbound") {
    return handleInbound(req, res);
  }
  if (req.method !== "POST") {
    res.statusCode = 405;
    return res.end();
  }
  try {
    if (req.url === "/healthz") {
      return ok(res, { loggedIn });
    }
    if (req.url === "/shutdown") {
      ok(res, {});
      setTimeout(() => process.kill(process.pid, "SIGTERM"), 50);
      return;
    }
    const body = await readBody(req);
    if (req.url === "/send") {
      const { chatId, text } = body || {};
      if (!chatId || typeof text !== "string") {
        return badRequest(res, "chatId and text are required");
      }
      const target = await resolveSayTarget(chatId);
      const result = await target.say(text);
      const messageId =
        result && typeof result === "object" && result.id ? result.id : null;
      return ok(res, { messageId });
    }
    if (req.url === "/send-file") {
      const { chatId, path: filePath, name, caption } = body || {};
      if (!chatId || typeof filePath !== "string" || !filePath) {
        return badRequest(res, "chatId and path are required");
      }
      await fs.access(filePath);
      const fileBox = FileBox.fromFile(filePath, name || path.basename(filePath));
      const target = await resolveSayTarget(chatId);
      const result = await target.say(fileBox);
      if (caption && typeof caption === "string") {
        await target.say(caption);
      }
      const messageId =
        result && typeof result === "object" && result.id ? result.id : null;
      return ok(res, { messageId });
    }
    if (req.url === "/typing") {
      // Wechaty has no universal typing API across puppets — no-op success.
      return ok(res, {});
    }
    res.statusCode = 404;
    res.setHeader("Content-Type", "application/json");
    return res.end(JSON.stringify({ ok: false, error: "not found" }));
  } catch (e) {
    console.error(
      "wechaty-sidecar: handler error: " +
        (e && e.stack ? e.stack : String(e))
    );
    return serverError(res);
  }
});

server.listen(port, bind, () => {
  console.error(`wechaty-sidecar: listening on ${bind}:${port}`);
});

let stopping = false;
async function shutdown(signal) {
  if (stopping) return;
  stopping = true;
  console.error(`wechaty-sidecar: received ${signal}, stopping...`);
  try {
    await Promise.race([
      bot.stop(),
      new Promise((resolve) => setTimeout(resolve, 5000)),
    ]);
  } catch (e) {
    console.error("wechaty-sidecar: bot.stop() failed: " + String(e));
  }
  server.close(() => process.exit(0));
  setTimeout(() => process.exit(1), 500).unref();
}

process.on("SIGINT", () => shutdown("SIGINT"));
process.on("SIGTERM", () => shutdown("SIGTERM"));

if (process.env.WECHATY_SIDECAR_WATCH_STDIN === "1") {
  process.stdin.resume();
  process.stdin.on("end", () => shutdown("stdin EOF (parent exited)"));
  process.stdin.on("error", () => shutdown("stdin error (parent exited)"));
}

process.on("unhandledRejection", (reason) => {
  console.error(
    "wechaty-sidecar: unhandledRejection: " +
      (reason && reason.stack ? reason.stack : String(reason))
  );
});
