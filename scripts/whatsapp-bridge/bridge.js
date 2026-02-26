#!/usr/bin/env node
/**
 * WhatsApp Web bridge for Hermes Agent.
 *
 * Connects to WhatsApp via Baileys (no browser/Chromium needed) and exposes
 * HTTP endpoints consumed by the Python WhatsApp adapter
 * (gateway/platforms/whatsapp.py).
 *
 * Usage:
 *   node bridge.js --port 3000 --session /path/to/session
 *
 * Endpoints:
 *   GET  /messages       Poll for new incoming messages (returns & clears queue)
 *   POST /send           Send a message  { chatId, message, replyTo? }
 *   POST /typing         Send typing indicator  { chatId }
 *   GET  /chat/:chatId   Get chat info  { name, isGroup, participants }
 *   GET  /health         Health check
 *   GET  /media/:file    Serve downloaded media files
 */

import {
  makeWASocket,
  useMultiFileAuthState,
  DisconnectReason,
  downloadMediaMessage,
  fetchLatestBaileysVersion,
} from "@whiskeysockets/baileys";
import express from "express";
import fs from "fs";
import path from "path";
import pino from "pino";
import qrcode from "qrcode-terminal";

// ---------------------------------------------------------------------------
// CLI argument parsing
// ---------------------------------------------------------------------------

const args = process.argv.slice(2);

function getArg(name, defaultValue) {
  const idx = args.indexOf(`--${name}`);
  return idx !== -1 && args[idx + 1] ? args[idx + 1] : defaultValue;
}

const PORT = parseInt(getArg("port", "3000"), 10);
const SESSION_PATH = getArg(
  "session",
  path.join(process.env.HOME || "~", ".hermes", "whatsapp", "session")
);
const MEDIA_DIR = path.join(SESSION_PATH, "media");

fs.mkdirSync(SESSION_PATH, { recursive: true });
fs.mkdirSync(MEDIA_DIR, { recursive: true });

const logger = pino({ level: "warn" });

// ---------------------------------------------------------------------------
// Message queue (buffered for polling by the Python adapter)
// ---------------------------------------------------------------------------

const messageQueue = [];

// Recent messages stored for reply-quoting (limited to last 200)
const recentMessages = new Map();
const RECENT_MSG_LIMIT = 200;

function storeRecentMessage(msg) {
  if (recentMessages.size >= RECENT_MSG_LIMIT) {
    const oldest = recentMessages.keys().next().value;
    recentMessages.delete(oldest);
  }
  recentMessages.set(msg.key.id, msg);
}

// ---------------------------------------------------------------------------
// Group name cache
// ---------------------------------------------------------------------------

const groupNameCache = {};

async function getGroupName(sock, groupJid) {
  if (groupNameCache[groupJid]) return groupNameCache[groupJid];
  try {
    const meta = await sock.groupMetadata(groupJid);
    groupNameCache[groupJid] = meta.subject;
    return meta.subject;
  } catch {
    return groupJid;
  }
}

// ---------------------------------------------------------------------------
// Media helpers
// ---------------------------------------------------------------------------

const MIME_EXT = {
  "image/jpeg": ".jpg",
  "image/png": ".png",
  "image/webp": ".webp",
  "image/gif": ".gif",
  "video/mp4": ".mp4",
  "video/3gpp": ".3gp",
  "audio/ogg; codecs=opus": ".ogg",
  "audio/ogg": ".ogg",
  "audio/mp4": ".m4a",
  "audio/mpeg": ".mp3",
  "application/pdf": ".pdf",
};

const TYPE_EXT = {
  image: ".jpg",
  video: ".mp4",
  audio: ".ogg",
  ptt: ".ogg",
  document: ".bin",
  sticker: ".webp",
};

function getMediaExtension(mediaType, mimetype) {
  if (mimetype && MIME_EXT[mimetype]) return MIME_EXT[mimetype];
  return TYPE_EXT[mediaType] || ".bin";
}

function detectMediaType(msg) {
  if (msg.message?.imageMessage) return "image";
  if (msg.message?.videoMessage) return "video";
  if (msg.message?.stickerMessage) return "sticker";
  if (msg.message?.audioMessage) {
    return msg.message.audioMessage.ptt ? "ptt" : "audio";
  }
  if (msg.message?.documentMessage) return "document";
  if (msg.message?.documentWithCaptionMessage) return "document";
  return null;
}

function getMediaMessage(msg) {
  return (
    msg.message?.imageMessage ||
    msg.message?.videoMessage ||
    msg.message?.audioMessage ||
    msg.message?.stickerMessage ||
    msg.message?.documentMessage ||
    msg.message?.documentWithCaptionMessage?.message?.documentMessage ||
    null
  );
}

/**
 * Strip the WhatsApp JID suffix to get a human-readable name.
 * "xxxx@s.whatsapp.net" -> "xxxx"
 */
function jidToName(jid) {
  return jid.replace("@s.whatsapp.net", "").replace("@g.us", "");
}

// ---------------------------------------------------------------------------
// WhatsApp connection via Baileys
// ---------------------------------------------------------------------------

let sock = null;
let connectionState = "disconnected";

async function connectWhatsApp() {
  const { state, saveCreds } = await useMultiFileAuthState(SESSION_PATH);
  const { version } = await fetchLatestBaileysVersion();

  sock = makeWASocket({
    version,
    auth: state,
    logger,
    markOnlineOnConnect: true,
  });

  // Persist session credentials
  sock.ev.on("creds.update", saveCreds);

  // Connection status
  sock.ev.on("connection.update", ({ connection, lastDisconnect, qr }) => {
    if (qr) {
      console.log("\n[bridge] Scan this QR code with WhatsApp:\n");
      qrcode.generate(qr, { small: true });
      console.log("");
    }

    if (connection === "open") {
      connectionState = "connected";
      console.log(
        `[bridge] Connected to WhatsApp. Bridge ready on port ${PORT}.`
      );
    }

    if (connection === "close") {
      connectionState = "disconnected";
      const statusCode = lastDisconnect?.error?.output?.statusCode;

      if (statusCode === DisconnectReason.loggedOut) {
        console.error(
          "[bridge] Logged out. Delete the session directory and re-authenticate."
        );
        process.exit(1);
      }

      console.log("[bridge] Disconnected. Reconnecting in 3s...");
      setTimeout(connectWhatsApp, 3000);
    }
  });

  // Incoming messages
  sock.ev.on("messages.upsert", async ({ messages, type }) => {
    if (type !== "notify") return;

    for (const msg of messages) {
      // Skip our own messages
      if (msg.key.fromMe) continue;
      // Skip status/broadcast updates
      if (msg.key.remoteJid === "status@broadcast") continue;

      storeRecentMessage(msg);

      const chatId = msg.key.remoteJid;
      const isGroup = chatId.endsWith("@g.us");
      const senderId = isGroup ? msg.key.participant : chatId;
      const senderName =
        msg.pushName || jidToName(senderId || chatId) || "Unknown";
      const chatName = isGroup
        ? await getGroupName(sock, chatId)
        : senderName;

      // Extract text body from various message types
      const body =
        msg.message?.conversation ||
        msg.message?.extendedTextMessage?.text ||
        msg.message?.imageMessage?.caption ||
        msg.message?.videoMessage?.caption ||
        msg.message?.documentMessage?.caption ||
        msg.message?.documentWithCaptionMessage?.message?.documentMessage
          ?.caption ||
        "";

      const event = {
        messageId: msg.key.id,
        chatId,
        chatName,
        senderId,
        senderName,
        isGroup,
        body,
        hasMedia: false,
        mediaType: "",
        mediaUrls: [],
        timestamp:
          (msg.messageTimestamp || Math.floor(Date.now() / 1000)) * 1000,
      };

      // Handle media messages
      const mediaType = detectMediaType(msg);
      const mediaMsg = getMediaMessage(msg);

      if (mediaType && mediaMsg) {
        event.hasMedia = true;
        event.mediaType = mediaType;

        try {
          const buffer = await downloadMediaMessage(
            msg,
            "buffer",
            {},
            { logger, reuploadRequest: sock.updateMediaMessage }
          );
          const ext = getMediaExtension(mediaType, mediaMsg.mimetype);
          const filename = `${msg.key.id}${ext}`;
          const filepath = path.join(MEDIA_DIR, filename);
          fs.writeFileSync(filepath, buffer);
          event.mediaUrls = [`http://localhost:${PORT}/media/${filename}`];
        } catch (dlErr) {
          console.error("[bridge] Media download failed:", dlErr.message);
        }
      }

      messageQueue.push(event);
    }
  });
}

// ---------------------------------------------------------------------------
// Express HTTP server
// ---------------------------------------------------------------------------

const app = express();
app.use(express.json());
app.use("/media", express.static(MEDIA_DIR));

// Health check
app.get("/health", (_req, res) => {
  res.json({ status: connectionState });
});

// Poll messages - returns queued messages and clears the queue
app.get("/messages", (_req, res) => {
  const messages = messageQueue.splice(0);
  res.json(messages);
});

// Send message
app.post("/send", async (req, res) => {
  if (!sock || connectionState !== "connected") {
    return res.status(503).json({ error: "Not connected to WhatsApp" });
  }

  try {
    const { chatId, message, replyTo } = req.body;
    if (!chatId || !message) {
      return res
        .status(400)
        .json({ error: "chatId and message are required" });
    }

    const opts = {};
    if (replyTo) {
      const original = recentMessages.get(replyTo);
      if (original) {
        opts.quoted = original;
      }
    }

    const sent = await sock.sendMessage(chatId, { text: message }, opts);
    res.json({ messageId: sent.key.id });
  } catch (err) {
    console.error("[bridge] Send error:", err.message);
    res.status(500).json({ error: err.message });
  }
});

// Typing indicator
app.post("/typing", async (req, res) => {
  try {
    const { chatId } = req.body;
    if (chatId && sock && connectionState === "connected") {
      await sock.presenceSubscribe(chatId);
      await sock.sendPresenceUpdate("composing", chatId);
    }
    res.json({ ok: true });
  } catch {
    // Typing failures are non-critical
    res.json({ ok: true });
  }
});

// Chat info
app.get("/chat/:chatId", async (req, res) => {
  const chatId = req.params.chatId;
  const isGroup = chatId.endsWith("@g.us");

  if (!sock || connectionState !== "connected") {
    return res.json({ name: jidToName(chatId), isGroup, participants: [] });
  }

  try {
    if (isGroup) {
      const metadata = await sock.groupMetadata(chatId);
      res.json({
        name: metadata.subject,
        isGroup: true,
        participants: metadata.participants.map((p) => ({
          id: p.id,
          admin: p.admin || null,
        })),
      });
    } else {
      res.json({ name: jidToName(chatId), isGroup: false, participants: [] });
    }
  } catch {
    res.json({ name: jidToName(chatId), isGroup, participants: [] });
  }
});

// ---------------------------------------------------------------------------
// Graceful shutdown
// ---------------------------------------------------------------------------

function shutdown() {
  console.log("[bridge] Shutting down...");
  if (sock) {
    sock.end(undefined);
  }
  process.exit(0);
}

process.on("SIGINT", shutdown);
process.on("SIGTERM", shutdown);

// ---------------------------------------------------------------------------
// Start
// ---------------------------------------------------------------------------

app.listen(PORT, () => {
  console.log(`[bridge] HTTP server listening on port ${PORT}`);
  console.log(`[bridge] Session path: ${SESSION_PATH}`);
  console.log(`[bridge] Media directory: ${MEDIA_DIR}`);

  connectWhatsApp().catch((err) => {
    console.error("[bridge] Fatal error:", err);
    process.exit(1);
  });
});
