import assert from "node:assert/strict";
import crypto from "node:crypto";
import fs from "node:fs";
import net from "node:net";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { spawn } from "node:child_process";

const route = Object.freeze({
  ticket_ref: "ESN-0407",
  assignment_id: "assignment",
  attempt_id: "attempt",
  owner_identity: "owner",
  ticket_revision: 1,
  authority_event_sha256: "a".repeat(64),
  correlation_id: "correlation",
  route_nonce: "b".repeat(64),
});

async function runSender(response, { ttlMs = 100, noResponse = false } = {}) {
  const directory = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), "esn0407-sender-test-")));
  fs.chmodSync(directory, 0o700);
  const socketPath = path.join(directory, "channel.sock");
  const questionPath = path.join(directory, "question.json");
  const configPath = path.join(directory, "mcp.json");
  const rawPath = path.join(directory, "raw.json");
  const verdictPath = path.join(directory, "verdict.json");
  const question = {
    type: "question", ...route, authority_scope: "answer-only-no-changes",
    ttl_ms: ttlMs, question: "Return the fixed marker only.",
  };
  fs.writeFileSync(questionPath, JSON.stringify(question) + "\n", { mode: 0o600 });
  fs.writeFileSync(configPath, JSON.stringify({ mcpServers: { esn0407_channel: { env: {
    ESN0407_TICKET_REF: route.ticket_ref,
    ESN0407_ASSIGNMENT_ID: route.assignment_id,
    ESN0407_ATTEMPT_ID: route.attempt_id,
    ESN0407_OWNER_IDENTITY: route.owner_identity,
    ESN0407_TICKET_REVISION: String(route.ticket_revision),
    ESN0407_AUTHORITY_EVENT_SHA256: route.authority_event_sha256,
    ESN0407_CORRELATION_ID: route.correlation_id,
    ESN0407_ROUTE_NONCE: route.route_nonce,
    ESN0407_CHANNEL_SOCKET: socketPath,
  } } } }) + "\n", { mode: 0o600 });
  const server = net.createServer((socket) => socket.once("data", () => {
    if (!noResponse) socket.end(response);
  }));
  await new Promise((resolve, reject) => server.listen(socketPath, resolve).once("error", reject));
  fs.chmodSync(socketPath, 0o600);
  const child = spawn(process.execPath, [new URL("./send-once.mjs", import.meta.url).pathname], {
    env: {
      ...process.env,
      ESN0407_CHANNEL_SOCKET: socketPath,
      ESN0407_QUESTION_FILE: questionPath,
      ESN0407_MCP_CONFIG: configPath,
      ESN0407_RAW_RESPONSE_EVIDENCE: rawPath,
      ESN0407_RESPONSE_VERDICT: verdictPath,
    },
    stdio: ["ignore", "pipe", "pipe"],
  });
  let stdout = "";
  let stderr = "";
  child.stdout.on("data", (chunk) => { stdout += chunk; });
  child.stderr.on("data", (chunk) => { stderr += chunk; });
  const status = await new Promise((resolve) => child.once("exit", resolve));
  await new Promise((resolve) => server.close(resolve));
  const raw = JSON.parse(fs.readFileSync(rawPath, "utf8"));
  const verdict = JSON.parse(fs.readFileSync(verdictPath, "utf8"));
  const modes = [fs.statSync(rawPath).mode & 0o777, fs.statSync(verdictPath).mode & 0o777];
  fs.rmSync(directory, { recursive: true });
  return { status, stdout, stderr, raw, verdict, modes };
}

test("sender persists exact bounded response before accepting the marker", async () => {
  const line = Buffer.from(JSON.stringify({ type: "reply", correlation_id: route.correlation_id, answer: "ESN0407_CHANNEL_OK" }));
  const result = await runSender(Buffer.concat([line, Buffer.from("\n")]));
  assert.equal(result.status, 0);
  assert.equal(result.verdict.outcome, "exact_reply");
  assert.equal(result.raw.raw_base64, line.toString("base64"));
  assert.equal(result.raw.raw_sha256, crypto.createHash("sha256").update(line).digest("hex"));
  assert.deepEqual(result.modes, [0o600, 0o600]);
  assert.equal(result.stdout.includes(route.correlation_id), false);
});

test("sender preserves and classifies an expiry without printing raw content", async () => {
  const line = Buffer.from(JSON.stringify({ type: "expired", correlation_id: route.correlation_id, secret: "must-not-print" }));
  const result = await runSender(Buffer.concat([line, Buffer.from("\n")]));
  assert.equal(result.status, 3);
  assert.equal(result.verdict.outcome, "remote_expired");
  assert.equal(result.raw.raw_base64, line.toString("base64"));
  assert.equal(result.stdout.includes("must-not-print"), false);
  assert.equal(result.stderr.includes("must-not-print"), false);
});

test("sender refuses to classify an uncorrelated remote status", async () => {
  const result = await runSender(Buffer.from('{"type":"expired","correlation_id":"other"}\n'));
  assert.equal(result.status, 3);
  assert.equal(result.verdict.outcome, "invalid_response");
});

test("sender preserves malformed bytes as base64 and emits a safe verdict", async () => {
  const line = Buffer.from([0x1b, 0x5b, 0x33, 0x31, 0x6d, 0xff]);
  const result = await runSender(Buffer.concat([line, Buffer.from("\n")]));
  assert.equal(result.status, 3);
  assert.equal(result.verdict.outcome, "invalid_response");
  assert.equal(result.raw.raw_base64, line.toString("base64"));
  assert.equal(result.stdout.includes("\u001b"), false);
  assert.equal(result.stderr, "");
});

test("sender preserves only a bounded prefix of an oversized response", async () => {
  const line = Buffer.alloc(4097, 0x78);
  const result = await runSender(Buffer.concat([line, Buffer.from("\n")]));
  assert.equal(result.status, 3);
  assert.equal(result.verdict.outcome, "oversized_response");
  assert.equal(result.raw.truncated, true);
  assert.equal(result.raw.raw_bytes_captured, 4096);
  assert.ok(result.raw.raw_bytes_observed_at_least > 4096);
  assert.equal(Buffer.from(result.raw.raw_base64, "base64").length, 4096);
});

test("sender records local timeout separately from any remote response", async () => {
  const result = await runSender(null, { ttlMs: 5, noResponse: true });
  assert.equal(result.status, 4);
  assert.equal(result.raw.event, "local_timeout");
  assert.equal(result.raw.raw_bytes, 0);
  assert.equal(result.verdict.outcome, "local_timeout");
});

test("sender classifies a peer EOF with partial bytes separately from timeout", async () => {
  const line = Buffer.from('{"type":"expired"');
  const result = await runSender(line);
  assert.equal(result.status, 3);
  assert.equal(result.verdict.outcome, "partial_eof");
  assert.equal(result.raw.raw_base64, line.toString("base64"));
});

