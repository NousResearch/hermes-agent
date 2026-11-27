// Protocol fixtures are synthetic; sockets, atomic journals and hook dispatch are real.
import test from "node:test";
import assert from "node:assert/strict";
import { EventEmitter, once } from "node:events";
import { spawnSync } from "node:child_process";
import { createConnection, createServer } from "node:net";
import { mkdtempSync, mkdirSync, writeFileSync, readFileSync, lstatSync, chmodSync, existsSync, rmSync, readdirSync, symlinkSync, realpathSync, readlinkSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import extension from "../../optional-skills/autonomous-ai-agents/tmux-supervision/scripts/tmux_supervisor/omp_extension.ts";

const runId = "a".repeat(32);
const poison = "PRIVATE_PROMPT_TOOL_RESULT_ERROR_DO_NOT_PERSIST";

function fixture(t, options = {}) {
  const root = realpathSync(mkdtempSync(join(tmpdir(), "t-")));
  const run = join(root, runId);
  mkdirSync(run, { mode: 0o700 });
  const binding = { version: 2, mode: "tmux", adapter: "omp", run_id: runId, workspace: root, tmux_session: "test-omp",
    owner: { hermes_home: root, platform: "discord", session_key: "fixture-key",
      session_id: "fixture-hermes", chat_id: "fixture-chat", thread_id: "fixture-thread" }, created_at: 1 };
  const bindingPath = join(run, "binding.json");
  writeFileSync(bindingPath, JSON.stringify(binding), { mode: 0o600 });
  const hooks = new Map();
  const ctx = { mode: "tui", cwd: root, hasUI: true, pending: false, sessionId: "fixture-omp",
    hasPendingMessages() { return this.pending; }, sessionManager: { getSessionId: () => ctx.sessionId } };
  const invoke = async (name, value = {}) => {
    for (const fn of hooks.get(name) ?? []) await fn({ type: name, messages: [], ...value }, ctx);
  };
  const install = () => {
    const previous = process.env.OMP_HERMES_BINDING_FILE;
    process.env.OMP_HERMES_BINDING_FILE = bindingPath;
    try { extension({ on(name, fn) { hooks.set(name, [...(hooks.get(name) ?? []), fn]); } }); }
    finally {
      if (previous === undefined) delete process.env.OMP_HERMES_BINDING_FILE;
      else process.env.OMP_HERMES_BINDING_FILE = previous;
    }
  };
  const sockets = [];
  const connect = async (request = { version: 2, type: "observe", run_id: runId, after_seq: 0 }) => {
    const socket = createConnection(join(run, "bridge.sock"));
    sockets.push(socket);
    const signal = new EventEmitter();
    const lines = [];
    let buffer = "";
    let closed = false;
    socket.on("error", () => {});
    socket.on("data", chunk => {
      buffer += chunk;
      while (buffer.includes("\n")) {
        const end = buffer.indexOf("\n");
        lines.push(JSON.parse(buffer.slice(0, end)));
        buffer = buffer.slice(end + 1);
      }
      signal.emit("changed");
    });
    socket.on("close", () => { closed = true; signal.emit("changed"); });
    const wait = async condition => {
      const abort = AbortSignal.timeout(3000);
      while (!condition()) await once(signal, "changed", { signal: abort });
    };
    await once(socket, "connect");
    if (request !== null) socket.write(typeof request === "string" || Buffer.isBuffer(request) ? request : JSON.stringify(request) + "\n");
    return { socket, lines, wait, count: n => wait(() => lines.length >= n || closed).then(() => assert.ok(lines.length >= n, `expected ${n} lines, got ${lines.length}`)),
      closed: () => wait(() => closed) };
  };
  t.after(async () => {
    for (const socket of sockets) socket.destroy();
    await invoke("session_shutdown");
    await new Promise(resolve => setImmediate(resolve));
    rmSync(root, { recursive: true, force: true });
  });
  if (!options.deferInstall) install();
  return { root, run, binding, bindingPath, hooks, ctx, invoke, install, connect,
    journal: () => JSON.parse(readFileSync(join(run, "journal.json"), "utf8")) };
}

function observe(after_seq, epoch) {
  return { version: 2, type: "observe", run_id: runId, after_seq, ...(epoch === undefined ? {} : { epoch }) };
}

test("no binding is an actual no-op", () => {
  const previous = process.env.OMP_HERMES_BINDING_FILE;
  delete process.env.OMP_HERMES_BINDING_FILE;
  try { extension({ on() { assert.fail("must not register hooks"); } }); }
  finally { if (previous !== undefined) process.env.OMP_HERMES_BINDING_FILE = previous; }
});

test("TUI-only guard uses ctx.mode, not hasUI", async t => {
  const f = fixture(t);
  f.ctx.mode = "rpc";
  await f.invoke("session_start");
  assert.equal(existsSync(join(f.run, "bridge.sock")), false);
  assert.equal(existsSync(join(f.run, "journal.json")), false);
});

for (const platform of ["slack", "telegram"]) {
  test(`synthetic ${platform} binding preserves owner and observation schema`, async t => {
    const f = fixture(t, { deferInstall: true });
    f.binding.owner.platform = platform;
    f.binding.owner.thread_id = platform === "telegram" ? null : "fixture-thread";
    const encoded = JSON.stringify(f.binding);
    writeFileSync(f.bindingPath, encoded);
    f.install();
    await f.invoke("session_start");
    const c = await f.connect();
    await c.count(1);
    assert.deepEqual(Object.keys(c.lines[0]).sort(),
      ["version", "type", "run_id", "epoch", "app_session_id", "pid", "seq", "state"].sort());
    await f.invoke("agent_start");
    await f.invoke("agent_end", { messages: [{ role: "assistant", stopReason: "stop" }] });
    await c.count(3);
    assert.deepEqual(c.lines.slice(1), f.journal().events);
    assert.deepEqual(c.lines.slice(1).map(e => e.kind), ["started", "turn_settled"]);
    for (const event of c.lines.slice(1)) {
      assert.deepEqual(Object.keys(event).sort(),
        ["version", "type", "run_id", "epoch", "seq", "app_session_id", "kind", "at_ms"].sort());
    }
    assert.equal(readFileSync(f.bindingPath, "utf8"), encoded);
  });

  test(`synthetic host accepts a ${platform} binding`, t => {
    const f = fixture(t, { deferInstall: true });
    f.binding.owner.platform = platform;
    writeFileSync(f.bindingPath, JSON.stringify(f.binding));
    const child = spawnSync(process.execPath, ["--experimental-strip-types",
      fileURLToPath(new URL("./fixtures/omp_tui_extension_host.mjs", import.meta.url))], {
      env: { ...process.env, OMP_HERMES_BINDING_FILE: f.bindingPath },
      input: '{"type":"session_shutdown"}\n', timeout: 3000, encoding: "utf8",
    });
    assert.equal(child.error, undefined);
    assert.equal(child.status, 0, child.stderr);
    assert.equal(child.stdout, "READY\nACK\n");
    assert.equal(f.journal().state, "closed");
  });
}

test("platform validation matches the Python identifier contract", async t => {
  for (const platform of ["Custom-Platform", "gateway.adapter_1", "x".repeat(64)]) {
    const f = fixture(t, { deferInstall: true });
    f.binding.owner.platform = platform;
    writeFileSync(f.bindingPath, JSON.stringify(f.binding));
    f.install();
    await f.invoke("session_start");
    assert.equal(f.journal().state, "idle");
    assert.equal(JSON.parse(readFileSync(f.bindingPath, "utf8")).owner.platform, platform);
  }
  for (const platform of [undefined, null, false, 1, [], {}, "", "slack\n", "tele\u0000gram", "x\u007f", "x".repeat(65), "a/b", ".gateway", "a b"]) {
    const f = fixture(t, { deferInstall: true });
    f.binding.owner.platform = platform;
    writeFileSync(f.bindingPath, JSON.stringify(f.binding));
    f.install();
    await f.invoke("session_start");
    assert.equal(f.hooks.size, 0);
    assert.deepEqual(readdirSync(f.run), ["binding.json"]);
  }
});

test("fixtures canonicalize a symlinked temporary directory", async t => {
  const parent = fixture(t, { deferInstall: true });
  const alias = join(parent.root, "alias");
  symlinkSync(tmpdir(), alias);
  const previous = process.env.TMPDIR;
  let f;
  try {
    process.env.TMPDIR = alias;
    f = fixture(t);
  } finally {
    if (previous === undefined) delete process.env.TMPDIR;
    else process.env.TMPDIR = previous;
  }
  assert.equal(f.root, realpathSync(f.root));
  await f.invoke("session_start");
  assert.equal(f.journal().state, "idle");
});

test("private socket, sanitized durable completion, coalescing and continued turns", async t => {
  const f = fixture(t);
  await f.invoke("session_start");
  assert.equal(lstatSync(join(f.run, "bridge.sock")).mode & 0o777, 0o600);
  assert.equal(lstatSync(join(f.run, "journal.json")).mode & 0o777, 0o600);
  const c = await f.connect();
  await c.count(1);
  assert.equal(c.lines[0].state, "idle");
  assert.match(c.lines[0].epoch, /^[a-f0-9]{32}$/);
  await f.invoke("agent_start");
  await f.invoke("agent_end", { messages: [poison], willContinue: true });
  await f.invoke("agent_end", { messages: [poison] });
  assert.deepEqual(f.journal().events.map(e => e.kind), ["started"]);
  await f.invoke("agent_start");
  f.ctx.pending = true;
  await f.invoke("agent_end", { messages: [poison] });
  assert.deepEqual(f.journal().events.map(e => e.kind), ["started", "started"]);
  f.ctx.pending = false;
  await f.invoke("agent_end", { messages: [poison] });
  assert.equal(f.journal().seq, 2);
  await f.invoke("agent_start");
  await f.invoke("agent_end", { messages: [poison] });
  await f.invoke("agent_end", { messages: [poison] });
  await c.count(5);
  assert.deepEqual(c.lines.slice(1).map(e => e.kind), ["started", "started", "started", "turn_settled"]);
  assert.equal(f.journal().seq, 4);
  assert.equal(f.journal().state, "idle");
  await f.invoke("agent_start");
  await f.invoke("agent_end", { messages: [poison] });
  await c.count(7);
  assert.equal(f.journal().seq, 6);
  for (const event of f.journal().events) {
    assert.deepEqual(Object.keys(event).sort(), ["version", "type", "run_id", "epoch", "seq", "app_session_id", "kind", "at_ms"].sort());
  }
  assert.equal(readFileSync(join(f.run, "journal.json"), "utf8").includes(poison), false);
  assert.deepEqual(readdirSync(f.run).sort(), ["binding.json", "bridge.sock", "journal.json"]);
});

test("approval hooks are session-scoped and discard raw fields", async t => {
  const f = fixture(t);
  await f.invoke("session_start");
  await f.invoke("tool_approval_requested", { sessionId: "other", reason: poison });
  assert.equal(f.journal().seq, 0);
  await f.invoke("tool_approval_requested", { sessionId: f.ctx.sessionId, toolName: poison, toolCallId: poison, reason: poison });
  assert.deepEqual(f.journal().events.map(e => e.kind), ["needs_input"]);
  assert.equal(JSON.stringify(f.journal()).includes(poison), false);
});

test("a provider failure is a sanitized error, not a settled completion", async t => {
  const f = fixture(t);
  await f.invoke("session_start");
  await f.invoke("agent_start");
  await f.invoke("agent_end", { messages: [{ role: "assistant", stopReason: "error", errorMessage: poison }] });
  assert.equal(f.journal().events.at(-1).kind, "error");
  assert.equal(f.journal().state, "idle");
  assert.equal(JSON.stringify(f.journal()).includes(poison), false);
});

test("a successful later turn does not inherit an earlier failure", async t => {
  const f = fixture(t);
  await f.invoke("session_start");
  const failure = { role: "assistant", stopReason: "error", errorMessage: poison };
  await f.invoke("agent_start");
  await f.invoke("agent_end", { messages: [failure] });
  await f.invoke("agent_start");
  await f.invoke("agent_end", { messages: [failure, { role: "assistant", stopReason: "stop" }] });
  assert.deepEqual(f.journal().events.map(e => e.kind), ["started", "error", "started", "turn_settled"]);
});

test("reconnect replays only unseen events with stable epoch and identity", async t => {
  const f = fixture(t);
  await f.invoke("session_start");
  const first = await f.connect();
  await first.count(1);
  const epoch = first.lines[0].epoch;
  await f.invoke("agent_start");
  await first.count(2);
  first.socket.destroy();
  await first.closed();
  await f.invoke("agent_end");
  const second = await f.connect(observe(1, epoch));
  await second.count(2);
  assert.equal(second.lines[0].epoch, epoch);
  assert.deepEqual(second.lines.slice(1).map(e => [e.seq, e.kind, e.app_session_id]), [[2, "turn_settled", "fixture-omp"]]);
  await f.invoke("agent_start");
  await second.count(3);
  assert.equal(second.lines[2].seq, 3);
});

test("invalid subscriptions do not poison healthy observers or the journal", async t => {
  const f = fixture(t);
  await f.invoke("session_start");
  for (let i = 0; i < 66; i++) { await f.invoke("agent_start"); await f.invoke("agent_end"); }
  assert.equal(f.journal().events.length, 128);
  assert.equal(f.journal().events[0].seq, 5);
  const before = f.journal();
  const healthy = await f.connect(observe(before.seq, before.epoch));
  await healthy.count(1);
  const gap = await f.connect(observe(0));
  await gap.closed();
  assert.deepEqual(gap.lines.map(e => e.type), ["hello", "rejected"]);
  assert.equal(gap.lines[1].reason, "invalid_cursor");
  const mismatch = await f.connect(observe(f.journal().seq, "b".repeat(32)));
  await mismatch.closed();
  assert.equal(mismatch.lines.at(-1).reason, "epoch_mismatch");
  const future = await f.connect(observe(f.journal().seq + 10, f.journal().epoch));
  await future.closed();
  assert.equal(future.lines.at(-1).reason, "invalid_cursor");
  assert.deepEqual(f.journal(), before);
  await f.invoke("agent_start");
  await f.invoke("agent_end");
  await healthy.count(3);
  assert.deepEqual(healthy.lines.slice(1).map(e => e.kind), ["started", "turn_settled"]);
});

test("unauthorized commands, malformed, oversized and extra frames cannot steer or mutate", async t => {
  const f = fixture(t);
  await f.invoke("session_start");
  for (const input of [
    { ...observe(0), type: "cancel" }, { ...observe(0), prompt: poison },
    { ...observe(0), run_id: "b".repeat(32) }, { ...observe(0), after_seq: -1 },
    { ...observe(0), after_seq: 0.5 }, { ...observe(0), after_seq: "0" },
    { ...observe(0), epoch: 123 }, { ...observe(0), version: observe(0).version + 1 },
    { ...observe(0), version: 1 },
    "null\n", "bad-json\n", Buffer.alloc(4097, 120),
    JSON.stringify(observe(0)) + "\n" + JSON.stringify(observe(0)) + "\n",
  ]) {
    const c = await f.connect(input);
    await c.closed();
    assert.deepEqual(c.lines, []);
  }
  assert.equal(f.journal().seq, 0);
  const c = await f.connect();
  await c.count(1);
  c.socket.write(JSON.stringify({ type: "input", text: poison }) + "\n");
  await c.closed();
  assert.equal(f.journal().seq, 0);
});

test("explicit session switch revokes original identity and never rebinds", async t => {
  const f = fixture(t);
  await f.invoke("session_start");
  const c = await f.connect();
  await c.count(1);
  await f.invoke("agent_start");
  f.ctx.sessionId = "new-omp";
  await f.invoke("session_switch", { reason: "new", previousSessionFile: poison });
  await c.closed();
  assert.equal(c.lines.at(-1).kind, "session_revoked");
  await f.invoke("session_start");
  await f.invoke("agent_start");
  await f.invoke("agent_end");
  const journal = f.journal();
  assert.equal(journal.state, "revoked");
  assert.equal(journal.app_session_id, "fixture-omp");
  assert.equal(journal.seq, 2);
  const reconnect = await f.connect(observe(1, journal.epoch));
  await reconnect.closed();
  assert.equal(reconnect.lines.at(-1).kind, "session_revoked");
});

test("session identity drift also revokes without an explicit switch hook", async t => {
  const f = fixture(t);
  await f.invoke("session_start");
  f.ctx.sessionId = "forked";
  await f.invoke("agent_start");
  assert.deepEqual(f.journal().events.map(e => e.kind), ["session_revoked"]);
});

test("shutdown is durable before closing observation", async t => {
  const f = fixture(t);
  await f.invoke("session_start");
  const c = await f.connect();
  await c.count(1);
  await f.invoke("session_shutdown");
  await c.closed();
  assert.equal(c.lines.at(-1).kind, "shutdown");
  assert.deepEqual(c.lines.at(-1), f.journal().events.at(-1));
  assert.equal(f.journal().state, "closed");
});

test("persistence failure closes clients without publishing an uncommitted completion", async t => {
  const f = fixture(t);
  await f.invoke("session_start");
  const c = await f.connect();
  await c.count(1);
  await f.invoke("agent_start");
  await c.count(2);
  rmSync(join(f.run, "journal.json"));
  mkdirSync(join(f.run, "journal.json"), { mode: 0o700 });
  await assert.doesNotReject(f.invoke("agent_end", { messages: [poison] }));
  await c.closed();
  assert.deepEqual(c.lines.slice(1).map(e => e.kind), ["started"]);
  assert.equal(readdirSync(f.run).some(name => name.startsWith(".journal-")), false);
  await assert.doesNotReject(f.invoke("agent_start"));
});

test("rejects loose permissions and symlink bindings", async t => {
  for (const mutation of [
    f => chmodSync(f.bindingPath, 0o644),
    f => chmodSync(f.run, 0o755),
    f => { const target = join(f.root, "binding-copy.json"); writeFileSync(target, JSON.stringify(f.binding), { mode: 0o600 }); rmSync(f.bindingPath); symlinkSync(target, f.bindingPath); },
  ]) {
    const f = fixture(t, { deferInstall: true });
    mutation(f);
    f.install();
    await f.invoke("session_start");
    assert.equal(f.hooks.size, 0);
    assert.equal(existsSync(join(f.run, "bridge.sock")), false);
  }
});

test("a FIFO binding is rejected without blocking the OMP process", t => {
  const f = fixture(t, { deferInstall: true });
  rmSync(f.bindingPath);
  assert.equal(spawnSync("mkfifo", ["-m", "600", f.bindingPath]).status, 0);
  const moduleUrl = new URL("../../optional-skills/autonomous-ai-agents/tmux-supervision/scripts/tmux_supervisor/omp_extension.ts", import.meta.url).href;
  const script = `import bridge from ${JSON.stringify(moduleUrl)}; bridge({on(){throw new Error("unexpected hooks")}});`;
  const child = spawnSync(process.execPath, ["--experimental-strip-types", "--input-type=module", "-e", script], {
    env: { ...process.env, OMP_HERMES_BINDING_FILE: f.bindingPath }, timeout: 1500, encoding: "utf8",
  });
  assert.equal(child.error, undefined);
  assert.equal(child.status, 0, child.stderr);
});

test("never reclaims an existing socket or overwrites a previous journal", async t => {
  const f = fixture(t);
  const incumbent = createServer(socket => socket.end("incumbent\n"));
  await new Promise(resolve => incumbent.listen(join(f.run, "bridge.sock"), resolve));
  const inode = lstatSync(join(f.run, "bridge.sock")).ino;
  await f.invoke("session_start");
  assert.equal(lstatSync(join(f.run, "bridge.sock")).ino, inode);
  assert.equal(existsSync(join(f.run, "journal.json")), false);
  await new Promise(resolve => incumbent.close(resolve));
  const g = fixture(t);
  writeFileSync(join(g.run, "journal.json"), "previous", { mode: 0o600 });
  await g.invoke("session_start");
  assert.equal(readFileSync(join(g.run, "journal.json"), "utf8"), "previous");
});

test("never replaces dangling socket or journal symlinks", async t => {
  for (const name of ["bridge.sock", "journal.json"]) {
    const f = fixture(t);
    const target = join(f.root, "missing");
    const occupied = join(f.run, name);
    symlinkSync(target, occupied);
    const inode = lstatSync(occupied).ino;
    await f.invoke("session_start");
    assert.equal(lstatSync(occupied).ino, inode);
    assert.equal(readlinkSync(occupied), target);
    assert.deepEqual(readdirSync(f.run).sort(), ["binding.json", name].sort());
  }
});

test("synthetic host does not announce readiness after refused startup", t => {
  const f = fixture(t, { deferInstall: true });
  writeFileSync(join(f.run, "journal.json"), "previous", { mode: 0o600 });
  const child = spawnSync(process.execPath, ["--experimental-strip-types",
    fileURLToPath(new URL("./fixtures/omp_tui_extension_host.mjs", import.meta.url))], {
    env: { ...process.env, OMP_HERMES_BINDING_FILE: f.bindingPath },
    input: "", timeout: 3000, encoding: "utf8",
  });
  assert.equal(child.error, undefined);
  assert.notEqual(child.status, 0);
  assert.equal(child.stdout, "");
  assert.equal(readFileSync(join(f.run, "journal.json"), "utf8"), "previous");
});

test("client count is bounded and partial requests are accepted within the size limit", async t => {
  const f = fixture(t);
  await f.invoke("session_start");
  const first = await f.connect(null);
  first.socket.write('{"version":2,"type":"observe",');
  first.socket.write(`"run_id":"${runId}","after_seq":0}\n`);
  await first.count(1);
  for (let i = 0; i < 31; i++) { const c = await f.connect(); await c.count(1); }
  const excess = await f.connect();
  await excess.closed();
  assert.deepEqual(excess.lines, []);
  assert.equal(f.journal().seq, 0);
});

test("a non-reading observer is disconnected without stalling OMP hooks", async t => {
  const f = fixture(t);
  await f.invoke("session_start");
  const slow = await f.connect();
  await slow.count(1);
  slow.socket.pause();
  // Fill the real socket and bounded queue without yielding to I/O.
  for (let i = 0; i < 1200; i++) await f.invoke("agent_start");
  assert.equal(f.journal().seq, 1200);
  assert.equal(f.journal().events.length, 128);
  slow.socket.resume();
  await slow.closed();
  const healthy = await f.connect(observe(1200, f.journal().epoch));
  await healthy.count(1);
  await f.invoke("agent_end");
  await healthy.count(2);
  assert.equal(healthy.lines[1].kind, "turn_settled");
});


test("OMP adapter rejects a generic command enrollment", async t => {
  const f = fixture(t, { deferInstall: true });
  f.binding.adapter = "command";
  writeFileSync(f.bindingPath, JSON.stringify(f.binding), { mode: 0o600 });
  f.install();
  await f.invoke("session_start");
  assert.equal(existsSync(join(f.run, "journal.json")), false);
  assert.equal(existsSync(join(f.run, "bridge.sock")), false);
});
