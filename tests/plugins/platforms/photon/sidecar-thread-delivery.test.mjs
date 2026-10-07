// Actual index.mjs HTTP handlers/helpers; SDK transport is networkless and recorded via IPC.
import test from "node:test";
import assert from "node:assert/strict";
import {fork} from "node:child_process";
import {once} from "node:events";
import net from "node:net";
import {fileURLToPath} from "node:url";

async function start(t) {
  const probe = net.createServer();
  probe.listen(0, "127.0.0.1");
  await once(probe, "listening");
  const port = probe.address().port;
  await new Promise(resolve => probe.close(resolve));
  const child = fork(fileURLToPath(new URL("../../../../plugins/platforms/photon/sidecar/index.mjs", import.meta.url)), [], {
    execArgv: ["--experimental-loader", fileURLToPath(new URL("./sidecar-sdk-loader.mjs", import.meta.url))],
    env: {...process.env, PHOTON_PROJECT_ID: "fixture", PHOTON_PROJECT_SECRET: "fixture",
      PHOTON_SIDECAR_TOKEN: "fixture", PHOTON_SIDECAR_PORT: String(port), PHOTON_SIDECAR_BIND: "127.0.0.1"},
    silent: true,
  });
  const events = [];
  child.on("message", message => events.push(message));
  let stderr = "";
  child.stderr.on("data", data => {stderr += data;});
  child.stdout.resume();
  t.after(async () => {child.kill("SIGTERM"); await once(child, "exit");});
  await new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error("sidecar startup timeout: " + stderr)), 5000);
    child.stderr.on("data", () => {if (stderr.includes("listening on")) {clearTimeout(timer); resolve();}});
    child.once("exit", code => {clearTimeout(timer); reject(new Error("sidecar exited " + code + ": " + stderr));});
  });
  async function post(route, body) {
    const response = await fetch(`http://127.0.0.1:${port}${route}`, {method: "POST",
      headers: {"x-hermes-sidecar-token": "fixture", "content-type": "application/json"}, body: JSON.stringify(body)});
    const bodyResult = await response.json();
    // HTTP and IPC use separate sockets: wait for an ordered IPC barrier so
    // assertions cannot race the recording messages from the completed handler.
    await new Promise((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error("IPC barrier timed out")), 5000);
      const listener = message => {
        if (message.type === "barrier") {
          clearTimeout(timer); child.off("message", listener); resolve();
        }
      };
      child.on("message", listener);
      child.send({type: "barrier"});
    });
    return {status: response.status, body: bodyResult};
  }
  return {post, events, port};
}

for (const kind of ["attachment", "voice"]) {
  test(`${kind} caption shares the resolved reply target`, async t => {
    const {post, events} = await start(t);
    const response = await post("/send-attachment", {spaceId: "space", path: "image.png", kind, caption: "caption", replyToId: "anchor"});
    assert.equal(response.status, 200);
    const sends = events.filter(e => e.type === "send").map(e => e.builder);
    assert.equal(sends.length, 2);
    assert.deepEqual(sends.map(s => s.type), ["reply", "reply"]);
    assert.deepEqual(sends.map(s => s.target.id), ["anchor", "anchor"]);
    assert.equal(events.filter(e => e.type === "lookup").length, 1);
  });
}

for (const route of ["/send", "/send-attachment"]) {
  test(`${route} recovers a definitive cached stale-target rejection`, async t => {
    const {post, events} = await start(t);
    const response = await post(route, {spaceId: "space", text: "answer", path: "image.png", caption: "caption", replyToId: "stale"});
    assert.equal(response.status, 200);
    assert.equal(response.body.messageId, "sent");
    const sends = events.filter(e => e.type === "send").map(e => e.builder);
    assert.equal(events.filter(e => e.type === "lookup").length, 0, "inbound cached target is used");
    assert.deepEqual(sends.map(s => s.type), route === "/send" ? ["reply", "text"] : ["reply", "attachment", "text"]);
  });
}

test("caption rejection recovers only the caption, never the already-sent attachment", async t => {
  const {post, events} = await start(t);
  const response = await post("/send-attachment", {spaceId: "space", path: "image.png", caption: "reject-caption", replyToId: "anchor"});
  assert.equal(response.status, 200);
  const sends = events.filter(e => e.type === "send").map(e => e.builder);
  assert.deepEqual(sends.map(s => s.type), ["reply", "reply", "text"]);
  assert.equal(sends[2].text, "reject-caption");
});

for (const route of ["/send", "/send-attachment"]) {
  test(`${route} recovers SDK-skipped unsupported replies`, async t => {
    const {post, events} = await start(t);
    const response = await post(route, {spaceId: "space", text: "answer", path: "image.png", caption: "caption", replyToId: "unsupported"});
    assert.equal(response.status, 200);
    assert.equal(response.body.messageId, "sent");
    assert.deepEqual(events.filter(e => e.type === "send").map(e => e.builder.type),
      route === "/send" ? ["reply", "text"] : ["reply", "attachment", "text"]);
  });
}

for (const failure of ["timeout", "drop", "503", "spoof", "wrong-code"]) {
  test(`${failure} is never resent for text, media or caption`, async t => {
    const {post, events} = await start(t);
    let response = await post("/send", {spaceId: "space", text: failure, replyToId: "anchor"});
    assert.equal(response.status, 500);
    assert.equal(events.filter(e => e.type === "send").length, 1);
    response = await post("/send-attachment", {spaceId: "space", path: failure, caption: "not sent", replyToId: "anchor"});
    assert.equal(response.status, 500);
    assert.equal(events.filter(e => e.type === "send").length, 2);
    response = await post("/send-attachment", {spaceId: "space", path: "image.png", caption: failure, replyToId: "anchor"});
    assert.equal(response.status, 200, "attachment was sent even though its caption failed");
    const sends = events.filter(e => e.type === "send").map(e => e.builder);
    assert.equal(sends.length, 4);
    assert.ok(sends.every(s => s.type === "reply"));
  });
}
