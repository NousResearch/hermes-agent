#!/usr/bin/env node
// Non-sending compatibility smoke test for Hermes' Photon/Spectrum local sidecar.
//
// This intentionally does NOT create a Spectrum client, connect to Photon, or
// send through Messages.app. It catches the fragile update points that have
// broken before: exact dependency pins, package-lock skew, Spectrum provider
// import paths, content builders, and native mixed-attachment behavior.

import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const sidecarDir = path.dirname(fileURLToPath(import.meta.url));
const readJson = (name) => JSON.parse(fs.readFileSync(path.join(sidecarDir, name), "utf8"));
const pkg = readJson("package.json");
const lock = readJson("package-lock.json");

const spectrumPackages = [
  "@spectrum-ts/core",
  "@spectrum-ts/imessage-local",
  "spectrum-ts",
];

for (const name of spectrumPackages) {
  const version = pkg.dependencies?.[name];
  assert(version, `${name} must be listed in dependencies`);
  assert(
    /^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?$/.test(version),
    `${name} must be pinned to an exact version, got ${version}`,
  );
  const lockVersion = lock.packages?.[`node_modules/${name}`]?.version;
  assert.equal(
    lockVersion,
    version,
    `${name} package-lock version (${lockVersion}) must match package.json (${version})`,
  );
}

assert.equal(
  pkg.dependencies["@spectrum-ts/core"],
  pkg.dependencies["@spectrum-ts/imessage-local"],
  "@spectrum-ts/core and @spectrum-ts/imessage-local must stay in lockstep",
);
assert.equal(
  pkg.dependencies["spectrum-ts"],
  pkg.dependencies["@spectrum-ts/core"],
  "spectrum-ts umbrella and @spectrum-ts/core must stay in lockstep",
);

for (const [name, version] of Object.entries(pkg.overrides ?? {})) {
  assert(
    /^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?$/.test(version),
    `override ${name} must be pinned to an exact version, got ${version}`,
  );
}

const core = await import("@spectrum-ts/core");
const local = await import("@spectrum-ts/imessage-local");
const cloud = await import("spectrum-ts/providers/imessage");

for (const name of ["text", "attachment", "markdown", "richlink", "typing", "poll", "option"]) {
  assert.equal(typeof core[name], "function", `@spectrum-ts/core.${name} must exist`);
}
assert.equal(typeof local.localIMessage, "function", "local provider must expose localIMessage()");
assert.equal(typeof local.effect, "function", "local provider must expose effect()");
assert.equal(typeof cloud.effect, "function", "cloud provider must expose effect()");
assert.equal(typeof cloud.imessage, "function", "cloud iMessage provider import path must expose imessage()");
const localConfig = local.localIMessage.config();
const cloudConfig = cloud.imessage.config();
assert.equal(localConfig.__name, "local_imessage");
assert.equal(cloudConfig.__name, "imessage");

assert.deepEqual(await core.text("hello").build(), { type: "text", text: "hello" });
assert.deepEqual(await core.typing().build(), { type: "typing", state: "start" });
assert.deepEqual(await core.poll("Lunch?", [core.option("Pizza"), core.option("Tacos")]).build(), {
  type: "poll",
  title: "Lunch?",
  options: [{ title: "Pizza" }, { title: "Tacos" }],
});

// Exercise Spectrum's actual public provider definition with a fake local
// IMessageSDK watcher. This proves v12 preserves the text segments around an
// Apple object-replacement attachment marker without rewriting package files.
let watcherCallbacks;
const fakeClient = {
  async startWatching(callbacks) {
    watcherCallbacks = callbacks;
  },
  async stopWatching() {},
};
const iterable = localConfig.__definition.messages({ client: fakeClient });
const iterator = iterable[Symbol.asyncIterator]();
const firstPending = iterator.next();
for (let i = 0; i < 20 && !watcherCallbacks; i += 1) {
  await new Promise((resolve) => setImmediate(resolve));
}
assert(watcherCallbacks, "local provider did not start its watcher");
watcherCallbacks.onIncomingMessage({
  id: "message-1",
  chatId: "any;-;+15555550100",
  chatKind: "dm",
  participant: "+15555550100",
  createdAt: new Date("2026-07-27T12:00:00Z"),
  kind: "text",
  reaction: null,
  retractedAt: null,
  hasAttachments: true,
  isAudioMessage: false,
  threadRootMessageId: null,
  text: "before\uFFFCafter",
  attachments: [{
    id: "attachment-1",
    fileName: "photo.jpg",
    mimeType: "image/jpeg",
    sizeBytes: 123,
    localPath: null,
    uti: "public.jpeg",
  }],
});
const emitted = [];
for (let i = 0; i < 3; i += 1) {
  const result = i === 0 ? await firstPending : await iterator.next();
  assert.equal(result.done, false);
  emitted.push(result.value);
}
assert.deepEqual(
  emitted.map((message) => ({
    type: message.content.type,
    partIndex: message.partIndex,
    text: message.content.text,
  })),
  [
    { type: "text", partIndex: 0, text: "before" },
    { type: "attachment", partIndex: 1, text: undefined },
    { type: "text", partIndex: 2, text: "after" },
  ],
);
await iterator.return();

// The retired Hermes patch targeted the cloud provider's inbound mapper, so
// exercise that exact public provider definition too. The fake streams avoid
// network access while preserving Spectrum's real event-to-message pipeline.
const queueStream = (items = []) => {
  let waiting;
  let closed = false;
  return {
    [Symbol.asyncIterator]() { return this; },
    async next() {
      if (items.length > 0) return { done: false, value: items.shift() };
      if (closed) return { done: true, value: undefined };
      return await new Promise((resolve) => { waiting = resolve; });
    },
    async return() {
      closed = true;
      waiting?.({ done: true, value: undefined });
      return { done: true, value: undefined };
    },
    async close() { await this.return(); },
  };
};
const cloudEvent = {
  type: "message.received",
  sequence: 1,
  occurredAt: new Date("2026-07-27T12:00:00Z"),
  chatGuid: "any;-;+15555550100",
  message: {
    guid: "message-1",
    isFromMe: false,
    isAudioMessage: false,
    sender: { address: "+15555550100", service: "iMessage" },
    chatGuids: ["any;-;+15555550100"],
    content: {
      text: "before\uFFFCafter",
      attachments: [{
        guid: "attachment-1",
        fileName: "photo.jpg",
        mimeType: "image/jpeg",
        totalBytes: 123,
        uti: "public.jpeg",
      }],
    },
  },
};
const messageEvents = queueStream([cloudEvent]);
const pollEvents = queueStream();
const fakeCloudClient = {
  messages: { subscribeEvents: () => messageEvents },
  polls: { subscribeEvents: () => pollEvents },
  events: { async *catchUp() {} },
};
const cloudIterable = cloudConfig.__definition.messages({
  client: [{ phone: "shared", client: fakeCloudClient }],
  projectConfig: undefined,
});
const cloudIterator = cloudIterable[Symbol.asyncIterator]();
const cloudResult = await cloudIterator.next();
assert.equal(cloudResult.done, false);
assert.equal(cloudResult.value.content.type, "group");
assert.deepEqual(
  cloudResult.value.content.items.map((message) => ({
    type: message.content.type,
    partIndex: message.partIndex,
    text: message.content.text,
  })),
  [
    { type: "text", partIndex: 0, text: "before" },
    { type: "attachment", partIndex: 1, text: undefined },
    { type: "text", partIndex: 2, text: "after" },
  ],
);
await cloudIterator.return();

console.log(JSON.stringify({
  ok: true,
  spectrumVersion: pkg.dependencies["spectrum-ts"],
  checks: [
    "exact-pins",
    "lockfile-sync",
    "native-mixed-attachments",
    "core-builders",
    "local-provider-import",
    "cloud-provider-import",
    "provider-configs",
  ],
}));
