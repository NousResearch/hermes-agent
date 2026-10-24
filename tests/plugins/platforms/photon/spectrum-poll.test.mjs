import assert from "node:assert/strict";
import { appendFileSync, cpSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";
import { fileURLToPath, pathToFileURL } from "node:url";
import { patchSpectrumTs } from "../../../../plugins/platforms/photon/sidecar/patch-spectrum-mixed-attachments.mjs";

const sidecar = fileURLToPath(new URL("../../../../plugins/platforms/photon/sidecar/", import.meta.url));

test("the installed Spectrum SDK preserves sent poll choices through its live stream", { timeout: 30_000 }, async (t) => {
  const root = mkdtempSync(path.join(tmpdir(), "hermes-spectrum-poll-"));
  t.after(() => rmSync(root, { recursive: true, force: true }));
  // A missing installation must fail this test. Copy symlink targets as files
  // so patching cannot write through a symlink into a shared package cache.
  cpSync(path.join(sidecar, "node_modules"), path.join(root, "node_modules"), {
    recursive: true,
    dereference: true,
  });
  patchSpectrumTs(root);
  const chunk = path.join(root, "node_modules/@spectrum-ts/imessage/dist/index.js");
  // Expose the real bundled implementation only in the disposable copy.
  appendFileSync(chunk, "\nexport { sendContent, messages$1, toCachedPoll };\n");
  const sdk = await import(pathToFileURL(chunk).href);

  const pollId = "test-poll";
  const chat = "any;-;+15551234567";
  const phone = "+15559990000";
  const options = [
    { text: "Route", optionIdentifier: "choice-1" },
    { text: "Calendar", optionIdentifier: "choice-2" },
  ];
  const bare = options.map(({ text }) => ({ text }));
  let refreshes = 0;
  let finishEvents;
  const finished = new Promise((resolve) => { finishEvents = resolve; });
  function subscription(events = [], onFinished = () => {}) {
    let close;
    const closed = new Promise((resolve) => { close = resolve; });
    return {
      async *[Symbol.asyncIterator]() {
        yield* events;
        onFinished();
        await closed;
      },
      async close() { close(); },
    };
  }
  const event = (sequence, delta) => ({
    type: "poll.changed",
    pollMessageGuid: pollId,
    chatGuid: chat,
    actor: { address: "+15551234567", service: "iMessage" },
    isFromMe: false,
    occurredAt: new Date(sequence * 1000),
    sequence,
    delta,
  });
  const remote = {
    polls: {
      async create(target, title, choices) {
        assert.equal(target, chat);
        assert.equal(title, "Which?");
        assert.deepEqual(choices, options.map(({ text }) => text));
        return { pollMessageGuid: pollId, title: "", options };
      },
      async get(id) {
        assert.equal(id, pollId);
        refreshes++;
        return { pollMessageGuid: pollId, title: "", options: bare };
      },
      subscribeEvents: () => subscription([
        event(1, { type: "created", title: "", options: bare }),
        // An unknown identifier forces a degraded API refresh before the real vote.
        event(2, { type: "voted", optionIdentifier: "unknown" }),
        event(3, { type: "voted", optionIdentifier: "choice-1" }),
        event(4, { type: "unvoted", optionIdentifier: "choice-1" }),
      ], finishEvents),
    },
    messages: { subscribeEvents: () => subscription() },
    groups: { subscribeEvents: () => subscription() },
  };
  const sent = await sdk.sendContent(remote, chat, chat, {
    type: "poll",
    title: "Which?",
    options: options.map(({ text }) => ({ title: text })),
  });
  assert.equal(sent.id, pollId);

  // Use messages$1 rather than constructing a cache ourselves: the live stream
  // must share the cache seeded by sendContent for this exact Photon client.
  const stream = sdk.messages$1([{ client: remote, phone }], {});
  const observed = [];
  const reader = (async () => {
    for await (const message of stream) observed.push(message.content);
  })();
  t.signal.addEventListener("abort", () => { void stream.close(); }, { once: true });
  try {
    await finished;
  } finally {
    await stream.close();
    await reader;
  }
  assert.deepEqual(observed.map(({ type, option, poll, selected }) => ({
    type, choice: option.title, question: poll.title, selected,
  })), [
    { type: "poll_option", choice: "Route", question: "Which?", selected: true },
    { type: "poll_option", choice: "Route", question: "Which?", selected: false },
  ]);
  assert.equal(refreshes, 1);

  // Empty inbound titles must also validate without an outbound cache entry.
  const inbound = sdk.toCachedPoll({ title: "", options });
  assert.ok(inbound.poll.title);
  assert.equal(inbound.optionsByIdentifier.get("choice-2").title, "Calendar");
});
