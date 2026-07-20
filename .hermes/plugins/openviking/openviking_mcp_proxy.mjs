#!/usr/bin/env node
/**
 * Minimal Hermes OpenViking MCP proxy.
 * stdio -> streamable-HTTP to OpenViking /mcp endpoint.
 * Reads OPENVIKING_URL, OPENVIKING_API_KEY, OPENVIKING_ACCOUNT, OPENVIKING_USER, OPENVIKING_ACTOR_PEER_ID.
 */
import { createInterface } from "node:readline";

const DEFAULT_PROTOCOL_VERSION = "2025-06-18";
const DEFAULT_TIMEOUT_MS = 15000;

function env(name, fallback = "") {
  return process.env[name] || fallback;
}

function trimSlash(v) {
  return String(v || "").replace(/\/+$/, "");
}

function errorResponse(id, code, message, data) {
  const err = { code, message };
  if (data !== undefined) err.data = data;
  return { jsonrpc: "2.0", id: id ?? null, error: err };
}

function parseJson(text) {
  const t = String(text || "").trim();
  if (!t) return null;
  try { return JSON.parse(t); } catch { return null; }
}

function parseSseMessages(text) {
  const messages = [];
  let dataLines = [];
  function flush() {
    if (dataLines.length === 0) return;
    const data = dataLines.join("\n").trim();
    dataLines = [];
    if (!data || data === "[DONE]") return;
    try { messages.push(JSON.parse(data)); } catch {}
  }
  for (const raw of String(text || "").split(/\r?\n/)) {
    if (raw === "") { flush(); continue; }
    if (raw.startsWith(":")) continue;
    const colon = raw.indexOf(":");
    const field = colon === -1 ? raw : raw.slice(0, colon);
    let value = colon === -1 ? "" : raw.slice(colon + 1);
    if (value.startsWith(" ")) value = value.slice(1);
    if (field === "data") dataLines.push(value);
  }
  flush();
  return messages;
}

function parseHttpBody(contentType, text) {
  const ctype = String(contentType || "").toLowerCase();
  if (!String(text || "").trim()) return [];
  if (ctype.includes("text/event-stream")) return parseSseMessages(text);
  const j = parseJson(text);
  return j == null ? [] : [j];
}

function headersForRequest(sessionId, cfg, protocolVersion) {
  const h = {
    "Content-Type": "application/json",
    "Accept": "application/json, text/event-stream",
    "MCP-Protocol-Version": protocolVersion,
  };
  if (sessionId) h["Mcp-Session-Id"] = sessionId;
  if (cfg.apiKey) h.Authorization = `Bearer ${cfg.apiKey}`;
  if (cfg.account) h["X-OpenViking-Account"] = cfg.account;
  if (cfg.user) h["X-OpenViking-User"] = cfg.user;
  if (cfg.peerId) h["X-OpenViking-Actor-Peer"] = cfg.peerId;
  return h;
}

function readConfig() {
  return {
    mcpUrl: `${trimSlash(env("OPENVIKING_URL", "http://127.0.0.1:1933"))}/mcp`,
    apiKey: env("OPENVIKING_API_KEY", ""),
    account: env("OPENVIKING_ACCOUNT", ""),
    user: env("OPENVIKING_USER", ""),
    peerId: env("OPENVIKING_ACTOR_PEER_ID", ""),
    timeoutMs: Math.max(1000, Number(env("OPENVIKING_TIMEOUT_MS", DEFAULT_TIMEOUT_MS))),
  };
}

async function main() {
  const cfg = readConfig();
  let sessionId = "";
  let protocolVersion = DEFAULT_PROTOCOL_VERSION;
  let stdoutChain = Promise.resolve();
  const rl = createInterface({ input: process.stdin, crlfDelay: Infinity });

  function writeMessage(obj) {
    const line = `${JSON.stringify(obj)}\n`;
    stdoutChain = stdoutChain
      .then(() => new Promise((resolve) => process.stdout.write(line, resolve)))
      .catch(() => {});
    return stdoutChain;
  }

  async function forwardToOpenViking(message) {
    const abort = new AbortController();
    const timer = setTimeout(() => abort.abort(), cfg.timeoutMs);
    try {
      const resp = await fetch(cfg.mcpUrl, {
        method: "POST",
        headers: headersForRequest(sessionId, cfg, protocolVersion),
        body: JSON.stringify(message),
        signal: abort.signal,
      });
      const bodyText = await resp.text();
      const contentType = resp.headers.get("content-type") || "";
      const newSessionId = resp.headers.get("mcp-session-id") || "";
      if (newSessionId) sessionId = newSessionId;
      const responses = parseHttpBody(contentType, bodyText);
      if (!resp.ok) {
        const detail = responses[0]?.error?.message || bodyText.slice(0, 200);
        return [errorResponse(
          message.id ?? null,
          -32001,
          `OpenViking MCP error ${resp.status}: ${detail}`,
          { status: resp.status }
        )];
      }
      if (responses.length === 0) {
        return [errorResponse(message.id ?? null, -32002, "Empty response from OpenViking")];
      }
      for (const r of responses) {
        if (r?.result?.sessionId) sessionId = r.result.sessionId;
        if (r?.result?.protocolVersion) protocolVersion = r.result.protocolVersion;
      }
      return responses;
    } catch (err) {
      return [errorResponse(
        message.id ?? null,
        -32003,
        `OpenViking MCP connection failed: ${err.message}`,
        { url: cfg.mcpUrl }
      )];
    } finally {
      clearTimeout(timer);
    }
  }

  for await (const line of rl) {
    const message = parseJson(line);
    if (!message || typeof message !== "object") continue;
    const responses = await forwardToOpenViking(message);
    for (const r of responses) await writeMessage(r);
  }
}

main().catch((err) => {
  console.error("OpenViking MCP proxy fatal error:", err);
  process.exit(1);
});
