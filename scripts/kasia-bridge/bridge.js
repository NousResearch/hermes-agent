#!/usr/bin/env node

import { createServer } from "node:http";
import { mkdir } from "node:fs/promises";
import { dirname, join } from "node:path";
import { homedir } from "node:os";

import { KasiaBridgeCore } from "./lib/bridge_core.js";

const args = process.argv.slice(2);

function getArg(name, defaultValue) {
  const flag = `--${name}`;
  const index = args.indexOf(flag);
  return index >= 0 && args[index + 1] ? args[index + 1] : defaultValue;
}

function resolveStateDir(value) {
  if (!value) {
    return join(homedir(), ".hermes", "kasia");
  }
  if (value.startsWith("~")) {
    return join(homedir(), value.slice(1));
  }
  return value;
}

function sendJson(res, statusCode, payload) {
  const body = `${JSON.stringify(payload)}\n`;
  res.writeHead(statusCode, {
    "Content-Type": "application/json",
    "Content-Length": Buffer.byteLength(body),
  });
  res.end(body);
}

async function readJsonBody(req) {
  const chunks = [];
  for await (const chunk of req) {
    chunks.push(Buffer.from(chunk));
  }
  if (chunks.length === 0) {
    return {};
  }
  return JSON.parse(Buffer.concat(chunks).toString("utf8"));
}

const port = Number.parseInt(getArg("port", "3010"), 10);
const stateDir = resolveStateDir(getArg("state-dir", ""));
const pollIntervalMs = Number.parseInt(getArg("poll-interval-ms", "4000"), 10);

const seedPhrase = process.env.KASIA_SEED_PHRASE || "";
const indexerUrl = process.env.KASIA_INDEXER_URL || "";
const nodeUrl = process.env.KASIA_NODE_WBORSH_URL || "";
const network = process.env.KASIA_NETWORK || "mainnet";
const feePolicy = process.env.KASIA_FEE_POLICY || "priority";
const maxMultipartParts = Number.parseInt(
  process.env.KASIA_MAX_MULTIPARTS || "8",
  10
);
const contextualMessageTargetChars = Number.parseInt(
  process.env.KASIA_TARGET_MESSAGE_CHARS || "240",
  10
);

if (!seedPhrase || !indexerUrl || !nodeUrl) {
  console.error(
    "KASIA_SEED_PHRASE, KASIA_INDEXER_URL, and KASIA_NODE_WBORSH_URL are required"
  );
  process.exit(1);
}

await mkdir(stateDir, { recursive: true });

const core = new KasiaBridgeCore({
  stateDir,
  indexerUrl,
  nodeUrl,
  network,
  seedPhrase,
  feePolicy,
  maxMultipartParts,
  contextualMessageTargetChars,
  logger: console,
});
await core.init();

let syncInFlight = null;
async function runSync() {
  if (syncInFlight) {
    return syncInFlight;
  }
  syncInFlight = core
    .syncOnce()
    .catch((error) => {
      console.warn(`[kasia-bridge] Sync failed: ${error?.message || error}`);
    })
    .finally(() => {
      syncInFlight = null;
    });
  return syncInFlight;
}

const server = createServer(async (req, res) => {
  try {
    const url = new URL(req.url || "/", "http://127.0.0.1");
    const method = req.method || "GET";

    if (method === "GET" && url.pathname === "/health") {
      return sendJson(res, 200, core.health());
    }

    if (method === "GET" && url.pathname === "/messages") {
      return sendJson(res, 200, core.dequeueMessages());
    }

    if (method === "POST" && url.pathname === "/handshakes/respond") {
      const body = await readJsonBody(req);
      const result = await core.respondToHandshake(body.chatId || body.address);
      return sendJson(res, 200, result);
    }

    if (method === "POST" && url.pathname === "/send") {
      const body = await readJsonBody(req);
      const result = await core.send({
        chatId: body.chatId,
        message: body.message,
        waitMs: body.waitMs,
      });
      return sendJson(res, 200, result);
    }

    if (method === "GET" && url.pathname.startsWith("/send/")) {
      const jobId = decodeURIComponent(url.pathname.slice("/send/".length));
      const result = core.getSendJob(jobId);
      if (!result) {
        return sendJson(res, 404, { error: "Send job not found" });
      }
      return sendJson(res, 200, result);
    }

    if (method === "GET" && url.pathname.startsWith("/chat/")) {
      const chatId = decodeURIComponent(url.pathname.slice("/chat/".length));
      return sendJson(res, 200, core.getChatInfo(chatId));
    }

    return sendJson(res, 404, { error: "Not found" });
  } catch (error) {
    return sendJson(res, 500, {
      error: error?.message || String(error),
    });
  }
});

server.listen(port, "127.0.0.1", () => {
  console.log(`[kasia-bridge] Listening on http://127.0.0.1:${port}`);
  console.log(`[kasia-bridge] State directory: ${stateDir}`);
  console.log(`[kasia-bridge] Wallet: ${core.health().walletAddress}`);
});

const syncTimer = setInterval(() => {
  runSync().catch(() => {});
}, pollIntervalMs);

async function shutdown(signalName) {
  clearInterval(syncTimer);
  server.close();
  await core.close();
  console.log(`[kasia-bridge] Stopped (${signalName})`);
  process.exit(0);
}

process.on("SIGINT", () => {
  shutdown("SIGINT").catch((error) => {
    console.error(error);
    process.exit(1);
  });
});
process.on("SIGTERM", () => {
  shutdown("SIGTERM").catch((error) => {
    console.error(error);
    process.exit(1);
  });
});
