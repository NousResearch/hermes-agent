// Hermes Agent - Gossip sidecar.
//
// Bridges Hermes' Python gateway adapter to @massalabs/gossip-sdk.
// Scope:
//   - text-only direct messages
//   - one configured admin user id
//   - no files, topics, or group chats
//   - edits/deletions are forwarded as explicit context events

import http from "node:http";
import { once } from "node:events";
import { dirname, join, resolve } from "node:path";
import { existsSync } from "node:fs";
import { fileURLToPath } from "node:url";

const createIdentityMode = process.argv.includes("--create-identity");
const mnemonic = (process.env.GOSSIP_MNEMONIC || "").trim();
const adminUserId = (process.env.GOSSIP_ADMIN_USER_ID || "").trim();
const storageDir = resolve(process.env.GOSSIP_STORAGE_DIR || "./gossip-storage");
const protocolBaseUrl = (process.env.GOSSIP_API_URL || "").trim();
const port = parseInt(process.env.GOSSIP_SIDECAR_PORT || "8797", 10);
const bind = process.env.GOSSIP_SIDECAR_BIND || "127.0.0.1";
const sharedToken = process.env.GOSSIP_SIDECAR_TOKEN;
const sidecarDir = dirname(fileURLToPath(import.meta.url));
const pollIntervalMs = Math.max(
  1000,
  parseInt(process.env.GOSSIP_POLL_INTERVAL_MS || "5000", 10) || 5000
);

if (!createIdentityMode && (!mnemonic || !adminUserId || !sharedToken)) {
  console.error(
    "gossip-sidecar: GOSSIP_MNEMONIC, GOSSIP_ADMIN_USER_ID and " +
      "GOSSIP_SIDECAR_TOKEN must all be set."
  );
  process.exit(2);
}

async function importGossipSdk() {
  const packageRoot = join(sidecarDir, "node_modules", "@massalabs", "gossip-sdk");
  const exportedDist = join(packageRoot, "dist", "index.js");
  if (!existsSync(packageRoot)) {
    console.error(
      "gossip-sidecar: @massalabs/gossip-sdk is not installed. Run:\n" +
        "  cd hermes-agent/plugins/platforms/gossip/sidecar && npm install"
    );
    process.exit(3);
  }
  if (!existsSync(exportedDist)) {
    console.error(
      "gossip-sidecar: @massalabs/gossip-sdk is installed, but its npm package " +
        "is missing dist/index.js. The package.json exports dist/index.js, so " +
        "Node cannot import it. Install a fixed npm release of " +
        "@massalabs/gossip-sdk that includes built dist files, then run setup again."
    );
    process.exit(3);
  }

  try {
    return await import("@massalabs/gossip-sdk");
  } catch (packageError) {
    console.error(
      "gossip-sidecar: failed to import @massalabs/gossip-sdk after resolving it. " +
        "Original import error: " +
        (packageError && packageError.stack ? packageError.stack : String(packageError))
    );
    process.exit(3);
  }
}

const {
  GossipSdk,
  SdkEventType,
  MessageDirection,
  MessageType,
  DiscussionDirection,
  generateMnemonic,
} = await importGossipSdk();

if (createIdentityMode) {
  try {
    const generatedMnemonic = generateMnemonic();
    const identitySdk = new GossipSdk();
    const initOptions = { storage: { type: "node-fs", path: storageDir } };
    if (protocolBaseUrl) initOptions.protocolBaseUrl = protocolBaseUrl;
    await identitySdk.init(initOptions);
    await identitySdk.openSession({
      mnemonic: generatedMnemonic,
      autoStartPolling: false,
    });
    const userId = identitySdk.userId;
    await identitySdk.closeSession();
    process.stdout.write(
      JSON.stringify({ ok: true, mnemonic: generatedMnemonic, userId }) + "\n"
    );
    process.exit(0);
  } catch (error) {
    process.stdout.write(
      JSON.stringify({
        ok: false,
        error: error && error.stack ? error.stack : String(error),
      }) + "\n"
    );
    process.exit(4);
  }
}

let sdk;
let ready = false;
let shuttingDown = false;
let consumerRes = null;
let consumerWaiters = [];
const seenEvents = new Map();
const recentMessages = new Map();
const MAX_SEEN = 4096;
const MAX_RECENT_MESSAGES = 2048;

function rememberEvent(key) {
  if (!key) return false;
  if (seenEvents.has(key)) return false;
  seenEvents.set(key, Date.now());
  if (seenEvents.size > MAX_SEEN) {
    const oldest = seenEvents.keys().next().value;
    if (oldest !== undefined) seenEvents.delete(oldest);
  }
  return true;
}

function waitForConsumer() {
  if (consumerRes) return Promise.resolve();
  return new Promise((resolveWait) => consumerWaiters.push(resolveWait));
}

function setConsumer(res) {
  consumerRes = res;
  const waiters = consumerWaiters;
  consumerWaiters = [];
  for (const resolveWait of waiters) resolveWait();
}

function clearConsumer(res) {
  if (consumerRes === res) consumerRes = null;
}

async function deliver(event) {
  const line = JSON.stringify(event) + "\n";
  while (!shuttingDown) {
    await waitForConsumer();
    const res = consumerRes;
    if (!res) continue;
    if (res.write(line)) return;
    await once(res, "drain").catch(() => {});
    if (consumerRes === res) return;
  }
}

function idToString(value) {
  if (value == null) return "";
  if (typeof value === "string" || typeof value === "number") return String(value);
  if (value instanceof Uint8Array) return Buffer.from(value).toString("base64url");
  if (Array.isArray(value)) return Buffer.from(value).toString("base64url");
  if (typeof value === "object" && typeof value.length === "number") {
    return Buffer.from(Array.from(value)).toString("base64url");
  }
  return String(value);
}

function rememberMessage(message) {
  const id = idToString(message?.messageId || message?.id);
  if (!id) return;
  recentMessages.set(id, {
    text: String(message.content || ""),
    contactUserId: message.contactUserId,
    direction: message.direction,
  });
  if (recentMessages.size > MAX_RECENT_MESSAGES) {
    const oldest = recentMessages.keys().next().value;
    if (oldest !== undefined) recentMessages.delete(oldest);
  }
}

function isIncomingAdminText(message) {
  if (!message) return false;
  if (message.contactUserId !== adminUserId) return false;
  if (message.direction !== MessageDirection.INCOMING && message.direction !== "incoming") {
    return false;
  }
  const type = message.type;
  return (
    type === MessageType.TEXT ||
    type === "text" ||
    type === MessageType.DELETED ||
    type === "deleted"
  );
}

function toEvent(kind, message) {
  const replyToId = idToString(message.replyTo?.originalMsgId);
  const replyTo = replyToId ? recentMessages.get(replyToId) : null;
  return {
    kind,
    contactUserId: message.contactUserId,
    dbId: message.id ?? null,
    messageId: idToString(message.messageId || message.id),
    replyToMessageId: replyToId || null,
    replyToText: replyTo?.text || null,
    text: String(message.content || ""),
    timestamp: message.timestamp
      ? new Date(message.timestamp).toISOString()
      : new Date().toISOString(),
  };
}

function forwardMessage(kind, message) {
  rememberMessage(message);
  if (!isIncomingAdminText(message)) return;
  const event = toEvent(kind, message);
  const key = `${kind}:${event.dbId || event.messageId}:${event.text}`;
  if (!rememberEvent(key)) return;
  void deliver(event).catch((error) => {
    console.error("gossip-sidecar: failed to deliver inbound event:", error);
  });
}

async function acceptAdminDiscussion(payload) {
  try {
    const discussion = payload?.discussion;
    const contact = payload?.contact;
    const contactUserId = discussion?.contactUserId || contact?.userId;
    if (contactUserId !== adminUserId) return;
    if (
      discussion?.direction &&
      discussion.direction !== DiscussionDirection.RECEIVED &&
      discussion.direction !== "received"
    ) {
      return;
    }
    await sdk.discussions.accept(discussion);
    await sdk.updateState();
    console.error("gossip-sidecar: accepted admin discussion request");
  } catch (error) {
    console.error(
      "gossip-sidecar: failed to accept admin discussion:",
      error && error.stack ? error.stack : String(error)
    );
  }
}

async function initializeSdk() {
  sdk = new GossipSdk();
  const initOptions = {
    storage: { type: "node-fs", path: storageDir },
    config: {
      polling: {
        enabled: true,
        messagesIntervalMs: pollIntervalMs,
        announcementsIntervalMs: pollIntervalMs,
        sessionRefreshIntervalMs: pollIntervalMs,
      },
    },
  };
  if (protocolBaseUrl) initOptions.protocolBaseUrl = protocolBaseUrl;

  await sdk.init(initOptions);
  sdk.on(SdkEventType.MESSAGE_RECEIVED, (message) => forwardMessage("message", message));
  sdk.on(SdkEventType.MESSAGE_SENT, (message) => rememberMessage(message));
  sdk.on(SdkEventType.MESSAGE_UPDATED, ({ messages }) => {
    for (const message of messages || []) forwardMessage("message_updated", message);
  });
  sdk.on(SdkEventType.MESSAGE_DELETED, ({ messages }) => {
    for (const message of messages || []) forwardMessage("message_deleted", message);
  });
  sdk.on(SdkEventType.SESSION_REQUESTED, acceptAdminDiscussion);
  sdk.on(SdkEventType.ERROR, ({ error, context }) => {
    console.error(
      `gossip-sidecar: SDK error in ${context}: ` +
        (error && error.stack ? error.stack : String(error))
    );
  });

  await sdk.openSession({ mnemonic });
  await sdk.announcements.fetch().catch(() => {});
  await sdk.updateState().catch(() => {});
  await sdk.messages.fetch().catch(() => {});

  ready = true;
  console.error(`gossip-sidecar: ready for admin ${adminUserId}`);
}

function readJson(req) {
  return new Promise((resolveRead, rejectRead) => {
    let body = "";
    req.setEncoding("utf8");
    req.on("data", (chunk) => {
      body += chunk;
      if (body.length > 1024 * 1024) {
        rejectRead(new Error("request body too large"));
        req.destroy();
      }
    });
    req.on("end", () => {
      if (!body.trim()) return resolveRead({});
      try {
        resolveRead(JSON.parse(body));
      } catch (error) {
        rejectRead(error);
      }
    });
    req.on("error", rejectRead);
  });
}

function sendJson(res, status, data) {
  res.writeHead(status, { "content-type": "application/json" });
  res.end(JSON.stringify(data));
}

function authorized(req) {
  return req.headers["x-hermes-sidecar-token"] === sharedToken;
}

const server = http.createServer(async (req, res) => {
  if (!authorized(req)) return sendJson(res, 401, { ok: false, error: "unauthorized" });

  try {
    if (req.method === "POST" && req.url === "/healthz") {
      return sendJson(res, ready ? 200 : 503, { ok: ready });
    }

    if (req.method === "GET" && req.url === "/inbound") {
      if (!ready) return sendJson(res, 503, { ok: false, error: "not ready" });
      res.writeHead(200, {
        "content-type": "application/x-ndjson",
        "cache-control": "no-cache",
        connection: "keep-alive",
      });
      setConsumer(res);
      const heartbeat = setInterval(() => {
        if (consumerRes === res) res.write("\n");
      }, 15000);
      req.on("close", () => {
        clearInterval(heartbeat);
        clearConsumer(res);
      });
      return;
    }

    if (req.method === "POST" && req.url === "/send") {
      if (!ready) return sendJson(res, 503, { ok: false, error: "not ready" });
      const body = await readJson(req);
      const contactUserId = String(body.contactUserId || "").trim();
      const text = String(body.text || "");
      if (contactUserId !== adminUserId) {
        return sendJson(res, 403, {
          ok: false,
          error: "only the configured admin can receive Gossip sends",
        });
      }
      if (!text.trim()) return sendJson(res, 400, { ok: false, error: "text is required" });

      const result = await sdk.messages.sendText(contactUserId, text, {
        metadata: { source: "hermes" },
      });
      if (!result.success) {
        return sendJson(res, 500, { ok: false, error: result.error || "sendText failed" });
      }
      await sdk.updateState();
      return sendJson(res, 200, {
        ok: true,
        messageId: idToString(result.message?.messageId || result.message?.id || ""),
      });
    }

    if (req.method === "POST" && req.url === "/shutdown") {
      sendJson(res, 200, { ok: true });
      shuttingDown = true;
      setTimeout(async () => {
        try {
          await sdk?.closeSession?.();
        } catch {}
        server.close(() => process.exit(0));
        setTimeout(() => process.exit(0), 1000).unref();
      }, 20).unref();
      return;
    }

    sendJson(res, 404, { ok: false, error: "not found" });
  } catch (error) {
    sendJson(res, 500, {
      ok: false,
      error: error && error.stack ? error.stack : String(error),
    });
  }
});

server.listen(port, bind, () => {
  console.error(`gossip-sidecar: listening on ${bind}:${port}`);
});

if (process.env.GOSSIP_SIDECAR_WATCH_STDIN === "1") {
  process.stdin.resume();
  process.stdin.on("end", () => process.exit(0));
  process.stdin.on("close", () => process.exit(0));
}

for (const sig of ["SIGINT", "SIGTERM"]) {
  process.on(sig, async () => {
    shuttingDown = true;
    try {
      await sdk?.closeSession?.();
    } catch {}
    process.exit(0);
  });
}

initializeSdk().catch((error) => {
  console.error(
    "gossip-sidecar: failed to initialize SDK:",
    error && error.stack ? error.stack : String(error)
  );
  process.exit(4);
});
