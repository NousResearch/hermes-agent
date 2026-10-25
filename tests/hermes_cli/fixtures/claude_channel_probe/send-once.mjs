#!/usr/bin/env node
import crypto from "node:crypto";
import fs from "node:fs";
import net from "node:net";
import path from "node:path";

import { MAX_ENVELOPE_BYTES, readExpectedRoute, validateQuestion } from "./contract.mjs";

const MARKER = "ESN0407_CHANNEL_OK";
const SCHEMA = "esn0407.channel-response-evidence/v1";

function requiredAbsolute(name) {
  const value = process.env[name];
  if (!value || !path.isAbsolute(value)) throw new Error(`${name} must be an absolute path`);
  return value;
}

function assertPrivateParent(filePath) {
  const parent = fs.realpathSync(path.dirname(filePath));
  const stat = fs.statSync(parent);
  if (!stat.isDirectory() || (stat.mode & 0o777) !== 0o700) {
    throw new Error("evidence parent must be a private mode-0700 directory");
  }
  if (typeof process.getuid === "function" && stat.uid !== process.getuid()) {
    throw new Error("evidence parent must be owned by this OS user");
  }
  return parent;
}

function readPrivateFile(filePath) {
  const before = fs.lstatSync(filePath);
  if (before.isSymbolicLink() || !before.isFile() || (before.mode & 0o777) !== 0o600) {
    throw new Error(`${filePath} must be a private mode-0600 regular file`);
  }
  if (typeof process.getuid === "function" && before.uid !== process.getuid()) {
    throw new Error(`${filePath} must be owned by this OS user`);
  }
  const descriptor = fs.openSync(filePath, fs.constants.O_RDONLY | (fs.constants.O_NOFOLLOW ?? 0));
  try {
    const after = fs.fstatSync(descriptor);
    if (before.dev !== after.dev || before.ino !== after.ino) throw new Error(`${filePath} changed while opening`);
    return fs.readFileSync(descriptor);
  } finally {
    fs.closeSync(descriptor);
  }
}

function writeExclusivePrivate(filePath, value) {
  const flags = fs.constants.O_WRONLY | fs.constants.O_CREAT | fs.constants.O_EXCL |
    (fs.constants.O_NOFOLLOW ?? 0);
  const descriptor = fs.openSync(filePath, flags, 0o600);
  try {
    fs.writeFileSync(descriptor, JSON.stringify(value) + "\n");
    fs.fsyncSync(descriptor);
  } finally {
    fs.closeSync(descriptor);
  }
}

const socketPath = requiredAbsolute("ESN0407_CHANNEL_SOCKET");
const questionPath = requiredAbsolute("ESN0407_QUESTION_FILE");
const configPath = requiredAbsolute("ESN0407_MCP_CONFIG");
const rawEvidencePath = requiredAbsolute("ESN0407_RAW_RESPONSE_EVIDENCE");
const verdictPath = requiredAbsolute("ESN0407_RESPONSE_VERDICT");
const evidenceParent = assertPrivateParent(rawEvidencePath);
if (assertPrivateParent(verdictPath) !== evidenceParent) throw new Error("evidence files must share one private parent");
if (fs.existsSync(rawEvidencePath) || fs.existsSync(verdictPath)) throw new Error("response evidence already exists");

const config = JSON.parse(readPrivateFile(configPath).toString("utf8"));
const routeEnvironment = config?.mcpServers?.esn0407_channel?.env;
const expectedRoute = readExpectedRoute(routeEnvironment ?? {});
if (routeEnvironment?.ESN0407_CHANNEL_SOCKET !== socketPath) {
  throw new Error("sender socket does not match the reviewed MCP route");
}
const questionBytes = readPrivateFile(questionPath);
if (questionBytes.length > MAX_ENVELOPE_BYTES) throw new Error("question envelope is too large");
const question = JSON.parse(questionBytes.toString("utf8"));
const ttlMs = validateQuestion(question, expectedRoute);

const socketStat = fs.lstatSync(socketPath);
if (!socketStat.isSocket() || (socketStat.mode & 0o777) !== 0o600) {
  throw new Error("reviewed private socket is not ready");
}
if (typeof process.getuid === "function" && socketStat.uid !== process.getuid()) {
  throw new Error("reviewed private socket is not owned by this OS user");
}

let settled = false;
let buffered = Buffer.alloc(0);
const client = net.createConnection(socketPath);
const timer = setTimeout(() => finishWithoutResponse("local_timeout", "LOCAL_TIMEOUT"), ttlMs + 5000);

function safeSummary(outcome, rawSha256 = null) {
  process.stdout.write(JSON.stringify({ schema: SCHEMA, outcome, raw_sha256: rawSha256 }) + "\n");
}

function finishWithoutResponse(outcome, code) {
  if (settled) return;
  settled = true;
  clearTimeout(timer);
  const evidence = { schema: SCHEMA, event: outcome, raw_bytes: 0, raw_sha256: null, raw_base64: null };
  writeExclusivePrivate(rawEvidencePath, evidence);
  writeExclusivePrivate(verdictPath, { schema: SCHEMA, outcome, raw_sha256: null, error_code: code });
  safeSummary(outcome);
  client.destroy();
  process.exitCode = 4;
}

function finishWithResponse(rawLine) {
  if (settled) return;
  settled = true;
  clearTimeout(timer);
  const rawSha256 = crypto.createHash("sha256").update(rawLine).digest("hex");
  writeExclusivePrivate(rawEvidencePath, {
    schema: SCHEMA,
    event: "remote_response_received_unvalidated",
    raw_bytes: rawLine.length,
    raw_sha256: rawSha256,
    raw_base64: rawLine.toString("base64"),
  });

  let outcome = "invalid_response";
  let errorCode = "INVALID_JSON_OR_SHAPE";
  try {
    const response = JSON.parse(rawLine.toString("utf8"));
    if (
      response?.type === "reply" &&
      response?.correlation_id === question.correlation_id &&
      response?.answer === MARKER
    ) {
      outcome = "exact_reply";
      errorCode = null;
    } else if (
      ["expired", "cancelled", "error"].includes(response?.type) &&
      response?.correlation_id === question.correlation_id
    ) {
      outcome = `remote_${response.type}`;
      errorCode = `REMOTE_${response.type.toUpperCase()}`;
    }
  } catch {}
  writeExclusivePrivate(verdictPath, {
    schema: SCHEMA,
    outcome,
    raw_sha256: rawSha256,
    error_code: errorCode,
  });
  safeSummary(outcome, rawSha256);
  client.end();
  process.exitCode = outcome === "exact_reply" ? 0 : 3;
}

function finishPartialEof(rawBytes) {
  if (settled) return;
  settled = true;
  clearTimeout(timer);
  const captured = rawBytes.subarray(0, MAX_ENVELOPE_BYTES);
  const rawSha256 = crypto.createHash("sha256").update(captured).digest("hex");
  writeExclusivePrivate(rawEvidencePath, {
    schema: SCHEMA,
    event: "partial_eof_unvalidated",
    raw_bytes: captured.length,
    raw_sha256: rawSha256,
    raw_base64: captured.toString("base64"),
  });
  writeExclusivePrivate(verdictPath, {
    schema: SCHEMA,
    outcome: "partial_eof",
    raw_sha256: rawSha256,
    error_code: "PARTIAL_EOF",
  });
  safeSummary("partial_eof", rawSha256);
  process.exitCode = 3;
}

function finishOversizedResponse(rawBytes) {
  if (settled) return;
  settled = true;
  clearTimeout(timer);
  const captured = rawBytes.subarray(0, MAX_ENVELOPE_BYTES);
  const rawSha256 = crypto.createHash("sha256").update(captured).digest("hex");
  writeExclusivePrivate(rawEvidencePath, {
    schema: SCHEMA,
    event: "remote_response_oversized_unvalidated",
    raw_bytes_captured: captured.length,
    raw_bytes_observed_at_least: rawBytes.length,
    truncated: true,
    raw_sha256: rawSha256,
    raw_base64: captured.toString("base64"),
  });
  writeExclusivePrivate(verdictPath, {
    schema: SCHEMA,
    outcome: "oversized_response",
    raw_sha256: rawSha256,
    error_code: "OVERSIZED_RESPONSE",
  });
  safeSummary("oversized_response", rawSha256);
  client.destroy();
  process.exitCode = 3;
}

client.once("connect", () => client.write(questionBytes.toString("utf8").trimEnd() + "\n"));
client.on("data", (chunk) => {
  if (settled) return;
  buffered = Buffer.concat([buffered, chunk]);
  if (buffered.length > MAX_ENVELOPE_BYTES) {
    finishOversizedResponse(buffered);
    return;
  }
  const newline = buffered.indexOf(0x0a);
  if (newline >= 0) finishWithResponse(buffered.subarray(0, newline));
});
client.once("end", () => {
  if (settled) return;
  if (buffered.length) finishPartialEof(buffered);
  else finishWithoutResponse("empty_eof", "EMPTY_EOF");
});
client.once("error", (error) => finishWithoutResponse("socket_error", String(error.code ?? "SOCKET_ERROR")));

