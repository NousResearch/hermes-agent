#!/usr/bin/env node

import express from "express";
import path from "path";
import qrcode from "qrcode-terminal";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "fs";
import { randomBytes, randomUUID } from "crypto";

const args = process.argv.slice(2);

function getArg(name, defaultValue) {
  const idx = args.indexOf(`--${name}`);
  return idx !== -1 && args[idx + 1] ? args[idx + 1] : defaultValue;
}

const PORT = Number.parseInt(getArg("port", "3010"), 10);
const SESSION_DIR = getArg(
  "session",
  path.join(process.env.HOME || "~", ".hermes", "weixin", "session"),
);
const BOT_TYPE = getArg("bot-type", process.env.WEIXIN_BOT_TYPE || "3");
const FIXED_BASE_URL = "https://ilinkai.weixin.qq.com";
const SESSION_EXPIRED_ERRCODE = -14;
const LONG_POLL_TIMEOUT_MS = 35_000;
const API_TIMEOUT_MS = 15_000;
const PACKAGE_VERSION = "0.1.0";
const ILINK_APP_ID = process.env.WEIXIN_APP_ID || "bot";

mkdirSync(SESSION_DIR, { recursive: true });

const messageQueue = [];
const MAX_QUEUE_SIZE = 200;

const state = {
  status: "starting",
  lastError: null,
  qrcodeUrl: null,
  credentials: null,
  contextTokens: new Map(),
  getUpdatesBuf: "",
  pollingBaseUrl: FIXED_BASE_URL,
};

let loginTask = null;
let monitorTask = null;
let shuttingDown = false;

function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function credentialsPath() {
  return path.join(SESSION_DIR, "credentials.json");
}

function contextTokensPath() {
  return path.join(SESSION_DIR, "context-tokens.json");
}

function syncBufPath() {
  return path.join(SESSION_DIR, "get-updates-buf.txt");
}

function readJson(filePath, fallback) {
  try {
    if (!existsSync(filePath)) return fallback;
    return JSON.parse(readFileSync(filePath, "utf8"));
  } catch {
    return fallback;
  }
}

function writeJson(filePath, value) {
  writeFileSync(filePath, JSON.stringify(value, null, 2), "utf8");
}

function loadStateFromDisk() {
  state.credentials = readJson(credentialsPath(), null);
  const persistedTokens = readJson(contextTokensPath(), {});
  state.contextTokens = new Map(
    Object.entries(persistedTokens).filter(([, token]) => typeof token === "string" && token),
  );
  try {
    if (existsSync(syncBufPath())) {
      state.getUpdatesBuf = readFileSync(syncBufPath(), "utf8").trim();
    }
  } catch {
    state.getUpdatesBuf = "";
  }
}

function persistContextTokens() {
  writeJson(contextTokensPath(), Object.fromEntries(state.contextTokens.entries()));
}

function persistCredentials() {
  writeJson(credentialsPath(), state.credentials);
}

function clearCredentials() {
  state.credentials = null;
  persistCredentials();
}

function buildClientVersion(version) {
  const parts = String(version)
    .split(".")
    .map((part) => Number.parseInt(part, 10) || 0);
  const major = parts[0] || 0;
  const minor = parts[1] || 0;
  const patch = parts[2] || 0;
  return ((major & 0xff) << 16) | ((minor & 0xff) << 8) | (patch & 0xff);
}

const ILINK_APP_CLIENT_VERSION = buildClientVersion(PACKAGE_VERSION);

function ensureTrailingSlash(url) {
  return url.endsWith("/") ? url : `${url}/`;
}

function randomWechatUin() {
  const uint32 = randomBytes(4).readUInt32BE(0);
  return Buffer.from(String(uint32), "utf8").toString("base64");
}

function buildCommonHeaders() {
  return {
    "iLink-App-Id": ILINK_APP_ID,
    "iLink-App-ClientVersion": String(ILINK_APP_CLIENT_VERSION),
  };
}

function buildPostHeaders(token, body) {
  const headers = {
    "Content-Type": "application/json",
    AuthorizationType: "ilink_bot_token",
    "Content-Length": String(Buffer.byteLength(body, "utf8")),
    "X-WECHAT-UIN": randomWechatUin(),
    ...buildCommonHeaders(),
  };
  if (token) {
    headers.Authorization = `Bearer ${token}`;
  }
  return headers;
}

async function apiGet(baseUrl, endpoint, timeoutMs = LONG_POLL_TIMEOUT_MS) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const url = new URL(endpoint, ensureTrailingSlash(baseUrl));
    const response = await fetch(url, {
      method: "GET",
      headers: buildCommonHeaders(),
      signal: controller.signal,
    });
    const rawText = await response.text();
    if (!response.ok) {
      throw new Error(`${endpoint} ${response.status}: ${rawText}`);
    }
    return rawText;
  } finally {
    clearTimeout(timer);
  }
}

async function apiPost(baseUrl, endpoint, payload, token, timeoutMs = API_TIMEOUT_MS) {
  const controller = new AbortController();
  const body = JSON.stringify(payload);
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const url = new URL(endpoint, ensureTrailingSlash(baseUrl));
    const response = await fetch(url, {
      method: "POST",
      headers: buildPostHeaders(token, body),
      body,
      signal: controller.signal,
    });
    const rawText = await response.text();
    if (!response.ok) {
      throw new Error(`${endpoint} ${response.status}: ${rawText}`);
    }
    return rawText;
  } finally {
    clearTimeout(timer);
  }
}

function buildBaseInfo() {
  return { channel_version: PACKAGE_VERSION };
}

function buildBodyFromItemList(itemList) {
  if (!Array.isArray(itemList)) return "";
  for (const item of itemList) {
    if (item?.type === 1 && item?.text_item?.text) {
      return String(item.text_item.text);
    }
    if (item?.type === 3 && item?.voice_item?.text) {
      return String(item.voice_item.text);
    }
  }
  return "";
}

function firstMediaType(itemList) {
  if (!Array.isArray(itemList)) return null;
  for (const item of itemList) {
    if (item?.type === 2) return "image";
    if (item?.type === 3) return "voice";
    if (item?.type === 4) return "document";
    if (item?.type === 5) return "video";
  }
  return null;
}

function pushMessage(event) {
  messageQueue.push(event);
  if (messageQueue.length > MAX_QUEUE_SIZE) {
    messageQueue.shift();
  }
}

async function fetchQRCode() {
  const rawText = await apiGet(
    FIXED_BASE_URL,
    `ilink/bot/get_bot_qrcode?bot_type=${encodeURIComponent(BOT_TYPE)}`,
    API_TIMEOUT_MS,
  );
  return JSON.parse(rawText);
}

async function pollQrStatus(baseUrl, qrcodeToken) {
  try {
    const rawText = await apiGet(
      baseUrl,
      `ilink/bot/get_qrcode_status?qrcode=${encodeURIComponent(qrcodeToken)}`,
      LONG_POLL_TIMEOUT_MS,
    );
    return JSON.parse(rawText);
  } catch (error) {
    if (error instanceof Error && error.name === "AbortError") {
      return { status: "wait" };
    }
    console.error("[weixin-bridge] QR polling error:", error.message || String(error));
    return { status: "wait" };
  }
}

async function beginLogin(force = false) {
  if (loginTask) {
    return loginTask;
  }

  loginTask = (async () => {
    let currentBaseUrl = FIXED_BASE_URL;
    let refreshCount = 0;
    while (!shuttingDown) {
      try {
        const qrResponse = await fetchQRCode();
        state.lastError = null;
        state.qrcodeUrl = qrResponse.qrcode_img_content || null;
        state.status = "qr_ready";
        console.log("[weixin-bridge] Scan this QR code with Weixin:");
        if (state.qrcodeUrl) {
          try {
            qrcode.generate(state.qrcodeUrl, { small: true });
          } catch {
            console.log(state.qrcodeUrl);
          }
        }

        while (!shuttingDown) {
          const statusResponse = await pollQrStatus(currentBaseUrl, qrResponse.qrcode);
          const qrStatus = statusResponse.status || "wait";
          if (qrStatus === "wait" || qrStatus === "scaned") {
            await delay(1000);
            continue;
          }
          if (qrStatus === "scaned_but_redirect" && statusResponse.redirect_host) {
            currentBaseUrl = `https://${statusResponse.redirect_host}`;
            await delay(500);
            continue;
          }
          if (qrStatus === "expired") {
            refreshCount += 1;
            if (refreshCount >= 3) {
              state.status = "needs_login";
              state.lastError = "QR code expired too many times";
              return;
            }
            console.log("[weixin-bridge] QR code expired, refreshing...");
            currentBaseUrl = FIXED_BASE_URL;
            break;
          }
          if (qrStatus === "confirmed") {
            state.credentials = {
              accountId: statusResponse.ilink_bot_id || randomUUID(),
              botToken: statusResponse.bot_token || "",
              baseUrl: statusResponse.baseurl || FIXED_BASE_URL,
              userId: statusResponse.ilink_user_id || "",
              connectedAt: new Date().toISOString(),
            };
            persistCredentials();
            state.qrcodeUrl = null;
            state.lastError = null;
            state.status = "connected";
            state.pollingBaseUrl = state.credentials.baseUrl || FIXED_BASE_URL;
            console.log(
              `[weixin-bridge] Login confirmed for ${state.credentials.accountId}`,
            );
            ensureMonitorLoop();
            return;
          }
          await delay(1000);
        }
      } catch (error) {
        state.status = "needs_login";
        state.lastError = error instanceof Error ? error.message : String(error);
        console.error("[weixin-bridge] Login flow error:", state.lastError);
        await delay(force ? 1000 : 3000);
      }
    }
  })().finally(() => {
    loginTask = null;
  });

  return loginTask;
}

async function getUpdates() {
  const credentials = state.credentials;
  if (!credentials?.botToken) {
    return { ret: 0, msgs: [], get_updates_buf: state.getUpdatesBuf };
  }
  try {
    const rawText = await apiPost(
      credentials.baseUrl || FIXED_BASE_URL,
      "ilink/bot/getupdates",
      {
        get_updates_buf: state.getUpdatesBuf || "",
        base_info: buildBaseInfo(),
      },
      credentials.botToken,
      LONG_POLL_TIMEOUT_MS,
    );
    return JSON.parse(rawText);
  } catch (error) {
    if (error instanceof Error && error.name === "AbortError") {
      return { ret: 0, msgs: [], get_updates_buf: state.getUpdatesBuf };
    }
    throw error;
  }
}

function normalizeInboundMessage(message) {
  if (!message?.from_user_id) return null;
  if (message?.message_type && Number(message.message_type) !== 1) return null;

  const mediaType = firstMediaType(message.item_list);
  let body = buildBodyFromItemList(message.item_list);
  if (!body && mediaType) {
    body = `[${mediaType} received]`;
  }

  const chatId = String(message.from_user_id);
  const contextToken = message.context_token;
  if (contextToken) {
    state.contextTokens.set(chatId, contextToken);
    persistContextTokens();
  }

  return {
    messageId: String(message.message_id || message.client_id || randomUUID()),
    chatId,
    senderId: chatId,
    senderName: chatId,
    chatName: chatId,
    isGroup: false,
    body,
    hasMedia: Boolean(mediaType),
    mediaType: mediaType || "",
    mediaUrls: [],
    mediaTypes: [],
    contextToken: contextToken || null,
    timestamp: message.create_time_ms || Date.now(),
  };
}

function ensureMonitorLoop() {
  if (monitorTask) return;
  monitorTask = (async () => {
    while (!shuttingDown) {
      if (!state.credentials?.botToken) {
        state.status = "needs_login";
        await delay(1000);
        continue;
      }
      try {
        const response = await getUpdates();
        const isApiError =
          (response.ret !== undefined && response.ret !== 0) ||
          (response.errcode !== undefined && response.errcode !== 0);
        if (isApiError) {
          if (
            response.errcode === SESSION_EXPIRED_ERRCODE ||
            response.ret === SESSION_EXPIRED_ERRCODE
          ) {
            console.warn("[weixin-bridge] Session expired, requesting re-login");
            clearCredentials();
            state.status = "needs_login";
            void beginLogin(true);
            await delay(2000);
            continue;
          }
          state.lastError = response.errmsg || `ret=${response.ret} errcode=${response.errcode}`;
          state.status = "error";
          await delay(2000);
          continue;
        }

        state.status = "connected";
        state.lastError = null;
        if (response.get_updates_buf) {
          state.getUpdatesBuf = String(response.get_updates_buf);
          writeFileSync(syncBufPath(), state.getUpdatesBuf, "utf8");
        }

        for (const message of response.msgs || []) {
          const normalized = normalizeInboundMessage(message);
          if (normalized) {
            pushMessage(normalized);
          }
        }
      } catch (error) {
        state.lastError = error instanceof Error ? error.message : String(error);
        console.error("[weixin-bridge] getUpdates failed:", state.lastError);
        await delay(2000);
      }
    }
  })().finally(() => {
    monitorTask = null;
  });
}

async function sendTextMessage(chatId, message, contextToken = undefined) {
  const credentials = state.credentials;
  if (!credentials?.botToken) {
    throw new Error("Weixin bridge is not paired yet");
  }
  const clientId = `hermes-weixin-${randomUUID()}`;
  await apiPost(
    credentials.baseUrl || FIXED_BASE_URL,
    "ilink/bot/sendmessage",
    {
      msg: {
        from_user_id: "",
        to_user_id: chatId,
        client_id: clientId,
        message_type: 2,
        message_state: 2,
        item_list: message ? [{ type: 1, text_item: { text: message } }] : undefined,
        context_token: contextToken,
      },
      base_info: buildBaseInfo(),
    },
    credentials.botToken,
    API_TIMEOUT_MS,
  );
  return { messageId: clientId };
}

const app = express();
app.use(express.json({ limit: "2mb" }));

app.get("/health", (_req, res) => {
  res.json({
    status: state.status,
    accountId: state.credentials?.accountId || null,
    userId: state.credentials?.userId || null,
    qrcodeUrl: state.qrcodeUrl,
    queueSize: messageQueue.length,
    lastError: state.lastError,
  });
});

app.get("/messages", (_req, res) => {
  const messages = messageQueue.splice(0, messageQueue.length);
  res.json(messages);
});

app.post("/send", async (req, res) => {
  const chatId = String(req.body?.chatId || "").trim();
  const message = String(req.body?.message || "");
  if (!chatId) {
    res.status(400).json({ error: "chatId is required" });
    return;
  }
  try {
    const contextToken = state.contextTokens.get(chatId);
    const result = await sendTextMessage(chatId, message, contextToken);
    res.json(result);
  } catch (error) {
    res.status(500).json({ error: error instanceof Error ? error.message : String(error) });
  }
});

app.post("/typing", (_req, res) => {
  res.json({ ok: true, noop: true });
});

app.get("/chat/:id", (req, res) => {
  const chatId = String(req.params.id || "");
  res.json({
    id: chatId,
    name: chatId,
    isGroup: false,
  });
});

const server = app.listen(PORT, () => {
  console.log(`[weixin-bridge] HTTP bridge listening on port ${PORT}`);
  loadStateFromDisk();
  if (state.credentials?.botToken) {
    state.status = "connected";
    ensureMonitorLoop();
  } else {
    state.status = "needs_login";
    void beginLogin();
  }
});

function shutdown() {
  shuttingDown = true;
  server.close(() => {
    process.exit(0);
  });
}

process.on("SIGINT", shutdown);
process.on("SIGTERM", shutdown);
