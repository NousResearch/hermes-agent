import assert from "node:assert/strict";
import fs from "node:fs";
import net from "node:net";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { spawnSync } from "node:child_process";

import {
  assertLockedDependencies,
  assertSubscriptionAuth,
  closeChannelResources,
  readExpectedRoute,
  validateQuestion,
  validateReply,
} from "./contract.mjs";

function environment(overrides = {}) {
  return {
    ESN0407_TICKET_REF: "ESN-0407",
    ESN0407_ASSIGNMENT_ID: "assignment-1",
    ESN0407_ATTEMPT_ID: "attempt-1",
    ESN0407_OWNER_IDENTITY: "operator",
    ESN0407_TICKET_REVISION: "4",
    ESN0407_AUTHORITY_EVENT_SHA256: "a".repeat(64),
    ESN0407_CORRELATION_ID: "correlation-1",
    ESN0407_ROUTE_NONCE: "b".repeat(64),
    ...overrides,
  };
}

function question(route, overrides = {}) {
  return {
    type: "question",
    authority_scope: "answer-only-no-changes",
    question: "Return the fixed marker only.",
    ttl_ms: 1000,
    ...route,
    ...overrides,
  };
}

test("route, question and reply bounds are enforced at runtime", () => {
  const route = readExpectedRoute(environment());
  assert.equal(validateQuestion(question(route), route), 1000);
  assert.deepEqual(
    validateReply({ correlation_id: route.correlation_id, answer: "OK" }, route),
    { correlationId: route.correlation_id, answer: "OK" },
  );
  assert.throws(() => readExpectedRoute(environment({ ESN0407_ROUTE_NONCE: "__NONCE__" })), /placeholder/);
  assert.throws(
    () => readExpectedRoute(environment({ ESN0407_CORRELATION_ID: "x".repeat(129) })),
    /too long/,
  );
  assert.throws(() => validateQuestion(question(route, { attempt_id: "other" }), route), /route mismatch/);
  assert.throws(
    () => validateReply({ correlation_id: "other", answer: "OK" }, route),
    /authorized route/,
  );
  assert.throws(
    () => validateReply({ correlation_id: route.correlation_id, answer: "x".repeat(2049) }, route),
    /answer too large/,
  );
});

test("preflight refuses provider credentials and non-subscription auth", () => {
  const subscription = {
    loggedIn: true,
    authMethod: "claude.ai",
    apiProvider: "firstParty",
    subscriptionType: "max",
  };
  assert.doesNotThrow(() => assertSubscriptionAuth(subscription, {}));
  assert.throws(() => assertSubscriptionAuth(subscription, { ANTHROPIC_API_KEY: "secret" }), /not scrubbed/);
  assert.throws(
    () => assertSubscriptionAuth({ ...subscription, authMethod: "apiKey" }, {}),
    /subscription authentication/,
  );
});

test("lockfile pins SDK integrity and every transitive", () => {
  const lock = JSON.parse(fs.readFileSync(new URL("./package-lock.json", import.meta.url)));
  assert.doesNotThrow(() => assertLockedDependencies(lock));
  const tampered = structuredClone(lock);
  tampered.packages["node_modules/@modelcontextprotocol/sdk"].integrity = "sha512-tampered";
  assert.throws(() => assertLockedDependencies(tampered), /integrity mismatch/);
});

test("teardown destroys active clients, clears timers and removes the socket", async () => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "esn0407-channel-test-"));
  fs.chmodSync(directory, 0o700);
  const socketPath = path.join(directory, "channel.sock");
  const activeSockets = new Set();
  const pending = new Map();
  const localServer = net.createServer((socket) => activeSockets.add(socket));
  await new Promise((resolve, reject) => {
    localServer.once("error", reject);
    localServer.listen(socketPath, resolve);
  });
  const client = net.createConnection(socketPath);
  await new Promise((resolve) => localServer.once("connection", resolve));
  const timer = setTimeout(() => {}, 60_000);
  pending.set("correlation-1", { socket: [...activeSockets][0], timer });

  await closeChannelResources({ localServer, activeSockets, pending, socketPath });

  assert.equal(pending.size, 0);
  assert.equal(fs.existsSync(socketPath), false);
  client.destroy();
  fs.rmSync(directory, { recursive: true });
});

test("launcher scrubs credentials, verifies subscription auth, and uses isolated flags", () => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "esn0407-launch-test-"));
  const bin = path.join(directory, "bin");
  const work = path.join(directory, "work");
  const config = path.join(directory, "mcp.json");
  const record = path.join(directory, "record.json");
  fs.mkdirSync(bin);
  fs.mkdirSync(work);
  fs.writeFileSync(config, "{}\n");
  const fakeClaude = `#!/bin/sh
if [ "\${1-}" = auth ]; then
  printf '%s\\n' '{"loggedIn":true,"authMethod":"claude.ai","apiProvider":"firstParty","subscriptionType":"max"}'
  exit 0
fi
node -e 'require("fs").writeFileSync(process.env.TEST_RECORD, JSON.stringify({args:process.argv.slice(1),api:process.env.ANTHROPIC_API_KEY || null,base:process.env.ANTHROPIC_BASE_URL || null,aws:process.env.AWS_PROFILE || null,cwd:process.cwd()}))' -- "$@"
`;
  fs.writeFileSync(path.join(bin, "claude"), fakeClaude, { mode: 0o755 });
  const result = spawnSync("sh", [new URL("./launch.sh", import.meta.url).pathname], {
    encoding: "utf8",
    env: {
      ...process.env,
      PATH: `${bin}:${process.env.PATH}`,
      TEST_RECORD: record,
      ESN0407_MCP_CONFIG: config,
      ESN0407_WORK_DIR: work,
      ANTHROPIC_API_KEY: "must-not-reach-child",
      ANTHROPIC_BASE_URL: "https://must-not-reach-child.invalid",
      AWS_PROFILE: "must-not-reach-child",
    },
  });
  assert.equal(result.status, 0, result.stdout + result.stderr);
  const observed = JSON.parse(fs.readFileSync(record));
  assert.deepEqual([observed.api, observed.base, observed.aws], [null, null, null]);
  assert.equal(observed.cwd, fs.realpathSync(work));
  assert.ok(observed.args.includes("--dangerously-load-development-channels"));
  assert.ok(observed.args.includes("--strict-mcp-config"));
  assert.ok(observed.args.includes("--setting-sources"));
  assert.ok(observed.args.includes(""));
  fs.rmSync(directory, { recursive: true });
});
