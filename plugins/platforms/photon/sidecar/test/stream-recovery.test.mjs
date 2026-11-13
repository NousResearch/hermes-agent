import assert from "node:assert/strict";
import { fork } from "node:child_process";
import { once } from "node:events";
import { test } from "node:test";

async function sidecar(t, extraEnv = {}) {
  const child = fork(new URL("./fixtures/pending-stream.mjs", import.meta.url), {
    execArgv: ["--experimental-test-module-mocks"],
    // Never inherit production credentials, profile paths or Node preloads.
    env: {
      PHOTON_PROJECT_ID: "test-project",
      PHOTON_PROJECT_SECRET: "test-secret",
      PHOTON_SIDECAR_TOKEN: "test-token",
      PHOTON_SIDECAR_PORT: "0",
      PHOTON_SIDECAR_BIND: "127.0.0.1",
      PHOTON_STREAM_DEGRADED_RESTART_MS: "1",
      ...extraEnv,
    },
    stdio: ["ignore", "ignore", "pipe", "ipc"],
  });
  let stderr = "";
  child.stderr.setEncoding("utf8").on("data", chunk => { stderr += chunk; });
  const exited = once(child, "exit");
  t.after(async () => {
    if (child.exitCode === null && child.signalCode === null) child.kill();
    await exited;
  });
  const [{ port }] = await once(child, "message", { signal: AbortSignal.timeout(5000) });
  return {
    exited,
    stderr: () => stderr,
    async warn(count) {
      const ack = once(child, "message", { signal: AbortSignal.timeout(5000) });
      child.send({ warnings: count });
      await ack;
    },
    async post(path) {
      const response = await fetch(`http://127.0.0.1:${port}${path}`, {
        method: "POST",
        headers: { "X-Hermes-Sidecar-Token": "test-token" },
        body: "{}",
        signal: AbortSignal.timeout(5000),
      });
      return { status: response.status, body: await response.json() };
    },
  };
}

test("the real telemetry WARN route reaches the stream classifier", { timeout: 10000 }, async t => {
  const app = await sidecar(t);
  await app.warn(1);
  const { body } = await app.post("/healthz");
  assert.match(app.stderr(), /\[spectrum.stream\] WARN stream interrupted; reconnecting/);
  assert.equal(body.stream.state, "recovering");
  assert.equal(body.stream.issueCount, 1);
  assert.equal(body.stream.ok, true);
});

test("three interrupted warnings with a pending iterator use the existing exit-75 recovery", { timeout: 10000 }, async t => {
  const app = await sidecar(t);
  await app.warn(3);
  const { body } = await app.post("/healthz");
  assert.equal(body.stream.state, "degraded");
  assert.equal(body.stream.ok, false);
  assert.equal(body.stream.lastHealthyAt, null);
  const [code, signal] = await app.exited;
  assert.equal(signal, null);
  assert.equal(code, 75);
  assert.match(app.stderr(), /exiting so Hermes can restart the Photon adapter/);
});

test("the probe reaches SDK GetMessage with a proxy message-resource GUID", { timeout: 10000 }, async t => {
  const app = await sidecar(t);
  const { status, body } = await app.post("/probe");
  assert.equal(status, 200);
  assert.equal(body.alive, true);
});

test("transport rejection stays inconclusive and does not degrade a quiet stream", { timeout: 10000 }, async t => {
  const app = await sidecar(t, { TEST_PROBE_FAILURE: "unavailable" });
  const { status, body } = await app.post("/probe");
  assert.equal(status, 503);
  assert.equal(body.outcome, "inconclusive");
  assert.equal(body.alive, false);
  assert.match(body.reason, /unavailable/i);
  const health = await app.post("/healthz");
  assert.equal(health.body.stream.state, "starting");
  assert.equal(health.body.stream.issueCount, 0);
});
