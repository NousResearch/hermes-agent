import assert from "node:assert/strict";
import fs from "node:fs";
import net from "node:net";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { spawn, spawnSync } from "node:child_process";

import {
  assertDisposableDirectory,
  assertLockedDependencies,
  assertManagedProfile,
  assertSubscriptionAuth,
  closeChannelResources,
  readExpectedRoute,
  validateQuestion,
  validateReply,
} from "./contract.mjs";
import { buildTrustBootstrapPlan } from "./trust-bootstrap-plan.mjs";

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

test("session preflight refuses a concurrently active managed-profile session", () => {
  const preflight = new URL("./preflight.mjs", import.meta.url).pathname;
  const idle = spawnSync("node", [preflight, "sessions"], { input: "[]", encoding: "utf8" });
  assert.equal(idle.status, 0, idle.stderr);
  const busy = spawnSync("node", [preflight, "sessions"], { input: '[{"id":"active"}]', encoding: "utf8" });
  assert.notEqual(busy.status, 0);
  assert.match(busy.stderr, /already has an active session/);
});

test("lockfile pins SDK integrity and every transitive", () => {
  const lock = JSON.parse(fs.readFileSync(new URL("./package-lock.json", import.meta.url)));
  assert.doesNotThrow(() => assertLockedDependencies(lock));
  const tampered = structuredClone(lock);
  tampered.packages["node_modules/@modelcontextprotocol/sdk"].integrity = "sha512-tampered";
  assert.throws(() => assertLockedDependencies(tampered), /integrity mismatch/);
});

test("managed profile binds private durable auth to one empty trusted workspace", () => {
  const profile = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), "esn0407-profile-test-")));
  fs.chmodSync(profile, 0o700);
  const config = path.join(profile, "config");
  const work = path.join(profile, "work");
  fs.mkdirSync(config, { mode: 0o700 });
  fs.mkdirSync(work, { mode: 0o700 });
  fs.writeFileSync(path.join(profile, "profile.json"), JSON.stringify({
    schema: "hermes.provider-auth-profile/v1", provider: "claude-code",
    account_kind: "claude.ai-subscription", config_directory: config,
    trusted_workspace: work, api_fallback_allowed: false,
    production_service_active: false,
  }), { mode: 0o600 });
  fs.writeFileSync(path.join(config, ".claude.json"), JSON.stringify({
    projects: { [work]: { hasTrustDialogAccepted: true } },
  }), { mode: 0o600 });
  assert.deepEqual(assertManagedProfile(profile, config, work), { profile, config, work });
  fs.chmodSync(work, 0o500);
  assert.throws(() => assertManagedProfile(profile, config, work), /mode-0700/);
  fs.chmodSync(work, 0o700);
  const claudeConfigPath = path.join(config, ".claude.json");
  const realConfigPath = path.join(config, "real-claude.json");
  fs.renameSync(claudeConfigPath, realConfigPath);
  fs.symlinkSync(realConfigPath, claudeConfigPath);
  assert.throws(() => assertManagedProfile(profile, config, work), /mode-0600/);
  fs.unlinkSync(claudeConfigPath);
  fs.renameSync(realConfigPath, claudeConfigPath);
  fs.writeFileSync(path.join(work, "unexpected"), "x");
  assert.throws(() => assertManagedProfile(profile, config, work), /must be empty/);
  fs.rmSync(profile, { recursive: true });
});

test("dependency preparation accepts only an owned private temporary non-repository", () => {
  const privateDirectory = fs.mkdtempSync(path.join(os.tmpdir(), "esn0407-location-test-"));
  fs.chmodSync(privateDirectory, 0o700);
  assert.equal(assertDisposableDirectory(privateDirectory), fs.realpathSync(privateDirectory));
  fs.chmodSync(privateDirectory, 0o755);
  assert.throws(() => assertDisposableDirectory(privateDirectory), /private mode 0700/);
  fs.chmodSync(privateDirectory, 0o700);
  fs.mkdirSync(path.join(privateDirectory, ".git"));
  assert.throws(() => assertDisposableDirectory(privateDirectory), /Git repository/);
  assert.throws(() => assertDisposableDirectory(new URL(".", import.meta.url).pathname), /temporary directory/);
  fs.rmSync(privateDirectory, { recursive: true });
});

test("trust bootstrap plan binds one empty private config to one empty private workspace", () => {
  const parent = fs.mkdtempSync(path.join(os.tmpdir(), "esn0407-trust-plan-test-"));
  fs.chmodSync(parent, 0o700);
  const config = path.join(parent, "config");
  const work = path.join(parent, "work");
  fs.mkdirSync(config, { mode: 0o700 });
  fs.mkdirSync(work, { mode: 0o700 });
  const plan = buildTrustBootstrapPlan(config, work);
  assert.equal(plan.target, path.join(fs.realpathSync(config), ".claude.json"));
  assert.deepEqual(plan.after.projects[fs.realpathSync(work)], { hasTrustDialogAccepted: true });
  assert.equal(plan.write_mode, "0600");
  assert.equal(fs.existsSync(plan.target), false);
  fs.writeFileSync(path.join(config, "unexpected"), "x");
  assert.throws(() => buildTrustBootstrapPlan(config, work), /must be empty/);
  fs.rmSync(parent, { recursive: true });
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
  const directory = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), "esn0407-launch-test-")));
  const bin = path.join(directory, "bin");
  const profile = path.join(directory, "profile");
  const claudeConfig = path.join(profile, "config");
  const work = path.join(profile, "work");
  const config = path.join(directory, "mcp.json");
  const record = path.join(directory, "record.json");
  fs.mkdirSync(bin);
  fs.mkdirSync(profile, { mode: 0o700 });
  fs.mkdirSync(claudeConfig, { mode: 0o700 });
  fs.mkdirSync(work, { mode: 0o700 });
  fs.writeFileSync(path.join(profile, "profile.json"), JSON.stringify({
    schema: "hermes.provider-auth-profile/v1", provider: "claude-code",
    account_kind: "claude.ai-subscription", config_directory: claudeConfig,
    trusted_workspace: work, api_fallback_allowed: false,
    production_service_active: false,
  }), { mode: 0o600 });
  fs.writeFileSync(path.join(claudeConfig, ".claude.json"), JSON.stringify({
    projects: { [work]: { hasTrustDialogAccepted: true } },
  }), { mode: 0o600 });
  fs.writeFileSync(config, "{}\n");
  const fakeClaude = `#!/bin/sh
if [ "\${1-}" = auth ]; then
  printf '%s\\n' '{"loggedIn":true,"authMethod":"claude.ai","apiProvider":"firstParty","subscriptionType":"max"}'
  exit 0
fi
if [ "\${1-}" = agents ]; then
  printf '%s\\n' '[]'
  exit 0
fi
IFS= read -r consent
TEST_STDIN="$consent" node -e 'const fs=require("fs"); fs.writeFileSync(process.env.TEST_RECORD, JSON.stringify({args:process.argv.slice(1),api:process.env.ANTHROPIC_API_KEY || null,base:process.env.ANTHROPIC_BASE_URL || null,aws:process.env.AWS_PROFILE || null,cwd:process.cwd(),stdin:process.env.TEST_STDIN,tty:Boolean(process.stdin.isTTY)}))' -- "$@"
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
      ESN0407_PROFILE_ROOT: profile,
      ESN0407_SESSION_ID: "12345678-1234-4abc-8def-1234567890ab",
      CLAUDE_CONFIG_DIR: claudeConfig,
      ANTHROPIC_API_KEY: "must-not-reach-child",
      ANTHROPIC_BASE_URL: "https://must-not-reach-child.invalid",
      AWS_PROFILE: "must-not-reach-child",
    },
    input: "operator-consent-input\n",
  });
  assert.equal(result.status, 0, result.stdout + result.stderr);
  const observed = JSON.parse(fs.readFileSync(record));
  assert.deepEqual([observed.api, observed.base, observed.aws], [null, null, null]);
  assert.equal(observed.cwd, fs.realpathSync(work));
  assert.equal(observed.stdin, "operator-consent-input");
  assert.equal(observed.tty, false);
  assert.equal(observed.args.includes("--bare"), false);
  assert.equal(observed.args.includes("--bg"), false);
  const sessionIndex = observed.args.indexOf("--session-id");
  assert.notEqual(sessionIndex, -1);
  assert.equal(observed.args[sessionIndex + 1], "12345678-1234-4abc-8def-1234567890ab");
  assert.ok(observed.args.includes("--dangerously-load-development-channels"));
  assert.ok(observed.args.includes("--strict-mcp-config"));
  const allowedIndex = observed.args.indexOf("--allowedTools");
  assert.notEqual(allowedIndex, -1);
  assert.equal(observed.args[allowedIndex + 1], "mcp__esn0407_channel__reply");
  assert.ok(observed.args.includes("--setting-sources"));
  assert.ok(observed.args.includes(""));
  assert.equal(fs.existsSync(path.join(profile, ".esn0407-channel.lock")), false);

  const ptyRunner = `
import os, pty, sys
pid, fd = pty.fork()
if pid == 0:
    os.execv('/bin/sh', ['/bin/sh', sys.argv[1]])
os.write(fd, b'operator-pty-consent\\n')
while True:
    try:
        if not os.read(fd, 4096):
            break
    except OSError:
        break
_, status = os.waitpid(pid, 0)
sys.exit(os.waitstatus_to_exitcode(status))
`;
  const ptyResult = spawnSync("python3", ["-c", ptyRunner, new URL("./launch.sh", import.meta.url).pathname], {
    encoding: "utf8",
    env: {
      ...process.env,
      PATH: `${bin}:${process.env.PATH}`,
      TEST_RECORD: record,
      ESN0407_MCP_CONFIG: config,
      ESN0407_WORK_DIR: work,
      ESN0407_PROFILE_ROOT: profile,
      ESN0407_SESSION_ID: "87654321-4321-4cba-9fed-ba0987654321",
      CLAUDE_CONFIG_DIR: claudeConfig,
    },
  });
  assert.equal(ptyResult.status, 0, ptyResult.stdout + ptyResult.stderr);
  const ptyObserved = JSON.parse(fs.readFileSync(record));
  assert.equal(ptyObserved.stdin, "operator-pty-consent");
  assert.equal(ptyObserved.tty, true);
  assert.equal(fs.existsSync(path.join(profile, ".esn0407-channel.lock")), false);
  fs.rmSync(directory, { recursive: true });
});

test("managed profile lease lets exactly one concurrent launcher reach Claude", async () => {
  const directory = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), "esn0407-race-test-")));
  const bin = path.join(directory, "bin");
  const profile = path.join(directory, "profile");
  const claudeConfig = path.join(profile, "config");
  const work = path.join(profile, "work");
  const mcpConfig = path.join(directory, "mcp.json");
  const record = path.join(directory, "record.txt");
  fs.mkdirSync(bin);
  fs.mkdirSync(profile, { mode: 0o700 });
  fs.mkdirSync(claudeConfig, { mode: 0o700 });
  fs.mkdirSync(work, { mode: 0o700 });
  fs.writeFileSync(path.join(profile, "profile.json"), JSON.stringify({
    schema: "hermes.provider-auth-profile/v1", provider: "claude-code",
    account_kind: "claude.ai-subscription", config_directory: claudeConfig,
    trusted_workspace: work, api_fallback_allowed: false,
    production_service_active: false,
  }), { mode: 0o600 });
  fs.writeFileSync(path.join(claudeConfig, ".claude.json"), JSON.stringify({
    projects: { [work]: { hasTrustDialogAccepted: true } },
  }), { mode: 0o600 });
  fs.writeFileSync(mcpConfig, "{}\n");
  fs.writeFileSync(path.join(bin, "claude"), `#!/bin/sh
if [ "\${1-}" = auth ]; then
  printf '%s\\n' '{"loggedIn":true,"authMethod":"claude.ai","apiProvider":"firstParty","subscriptionType":"max"}'
  exit 0
fi
if [ "\${1-}" = agents ]; then
  printf '%s\\n' '[]'
  exit 0
fi
printf '%s\\n' "$ESN0407_SESSION_ID" >> "$TEST_RECORD"
sleep 2
`, { mode: 0o755 });
  const baseEnv = {
    ...process.env, PATH: `${bin}:${process.env.PATH}`, TEST_RECORD: record,
    ESN0407_MCP_CONFIG: mcpConfig, ESN0407_WORK_DIR: work,
    ESN0407_PROFILE_ROOT: profile, CLAUDE_CONFIG_DIR: claudeConfig,
  };
  const launcher = new URL("./launch.sh", import.meta.url).pathname;
  const first = spawn("sh", [launcher], {
    env: { ...baseEnv, ESN0407_SESSION_ID: "12345678-1234-4abc-8def-1234567890ab" },
    stdio: ["ignore", "pipe", "pipe"],
  });
  const deadline = Date.now() + 1500;
  while (!fs.existsSync(record) && Date.now() < deadline) {
    await new Promise((resolve) => setTimeout(resolve, 10));
  }
  assert.equal(fs.existsSync(record), true, "first launcher did not reach Claude");
  const second = spawnSync("sh", [launcher], {
    env: { ...baseEnv, ESN0407_SESSION_ID: "87654321-4321-4cba-9fed-ba0987654321" },
    encoding: "utf8",
  });
  assert.notEqual(second.status, 0);
  assert.match(second.stderr, /already has a live launcher/);
  const firstStatus = await new Promise((resolve) => first.once("exit", resolve));
  assert.equal(firstStatus, 0);
  assert.deepEqual(fs.readFileSync(record, "utf8").trim().split("\n"), [
    "12345678-1234-4abc-8def-1234567890ab",
  ]);
  assert.equal(fs.existsSync(path.join(profile, ".esn0407-channel.lock")), false);
  fs.rmSync(directory, { recursive: true });
});

test("launcher signals cannot release the lease and continue or orphan Claude", async () => {
  const directory = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), "esn0407-signal-test-")));
  const bin = path.join(directory, "bin");
  const profile = path.join(directory, "profile");
  const claudeConfig = path.join(profile, "config");
  const work = path.join(profile, "work");
  const mcpConfig = path.join(directory, "mcp.json");
  const record = path.join(directory, "record.txt");
  fs.mkdirSync(bin);
  fs.mkdirSync(profile, { mode: 0o700 });
  fs.mkdirSync(claudeConfig, { mode: 0o700 });
  fs.mkdirSync(work, { mode: 0o700 });
  fs.writeFileSync(path.join(profile, "profile.json"), JSON.stringify({
    schema: "hermes.provider-auth-profile/v1", provider: "claude-code",
    account_kind: "claude.ai-subscription", config_directory: claudeConfig,
    trusted_workspace: work, api_fallback_allowed: false,
    production_service_active: false,
  }), { mode: 0o600 });
  fs.writeFileSync(path.join(claudeConfig, ".claude.json"), JSON.stringify({
    projects: { [work]: { hasTrustDialogAccepted: true } },
  }), { mode: 0o600 });
  fs.writeFileSync(mcpConfig, "{}\n");
  fs.writeFileSync(path.join(bin, "claude"), `#!/bin/sh
if [ "\${1-}" = auth ]; then
  printf '%s\\n' auth >> "$TEST_RECORD"
  [ -z "\${SLOW_AUTH-}" ] || sleep 5
  printf '%s\\n' '{"loggedIn":true,"authMethod":"claude.ai","apiProvider":"firstParty","subscriptionType":"max"}'
  exit 0
fi
if [ "\${1-}" = agents ]; then
  printf '%s\\n' agents >> "$TEST_RECORD"
  printf '%s\\n' '[]'
  exit 0
fi
exec node -e '
  const fs = require("fs");
  fs.appendFileSync(process.env.TEST_RECORD, "main\\n");
  process.on("SIGTERM", () => {
    fs.appendFileSync(process.env.TEST_RECORD, "child-term\\n");
    setTimeout(() => process.exit(0), 1000);
  });
  setInterval(() => {}, 1000);
'
`, { mode: 0o755 });
  const baseEnv = {
    ...process.env, PATH: `${bin}:${process.env.PATH}`, TEST_RECORD: record,
    ESN0407_MCP_CONFIG: mcpConfig, ESN0407_WORK_DIR: work,
    ESN0407_PROFILE_ROOT: profile, CLAUDE_CONFIG_DIR: claudeConfig,
    ESN0407_SESSION_ID: "12345678-1234-4abc-8def-1234567890ab",
  };
  const launcher = new URL("./launch.sh", import.meta.url).pathname;
  const waitFor = async (needle, timeout = 2000) => {
    const deadline = Date.now() + timeout;
    while (Date.now() < deadline) {
      if (fs.existsSync(record) && fs.readFileSync(record, "utf8").includes(needle)) return;
      await new Promise((resolve) => setTimeout(resolve, 10));
    }
    assert.fail(`timed out waiting for ${needle}`);
  };

  const duringAuth = spawn("sh", [launcher], {
    env: { ...baseEnv, SLOW_AUTH: "1" }, stdio: ["ignore", "pipe", "pipe"],
  });
  await waitFor("auth");
  duringAuth.kill("SIGTERM");
  const authStatus = await new Promise((resolve) => duringAuth.once("exit", resolve));
  assert.equal(authStatus, 143);
  assert.deepEqual(fs.readFileSync(record, "utf8").trim().split("\n"), ["auth"]);
  assert.equal(fs.existsSync(path.join(profile, ".esn0407-channel.lock")), false);

  fs.writeFileSync(record, "");
  const duringMain = spawn("sh", [launcher], {
    env: baseEnv, stdio: ["ignore", "pipe", "pipe"],
  });
  await waitFor("main");
  duringMain.kill("SIGTERM");
  await waitFor("child-term");
  assert.equal(fs.existsSync(path.join(profile, ".esn0407-channel.lock")), true);
  const mainStatus = await new Promise((resolve) => duringMain.once("exit", resolve));
  assert.equal(mainStatus, 143);
  assert.equal(fs.existsSync(path.join(profile, ".esn0407-channel.lock")), false);
  fs.rmSync(directory, { recursive: true });
});
